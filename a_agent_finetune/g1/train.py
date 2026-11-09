"""G1 single-GPU online GRPO; check materials, train from base, or evaluate an adapter."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .materials import load_rows, repository_for, verify_materials
from .metrics import TrainingMetrics
from .reward import DeliveryReward, binary_delivery_reward
from .rollout import Budgets, run_trajectory, save_trajectory

HERE = Path(__file__).resolve().parent


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def loading_report(info: dict) -> dict:
    """Transformers 5.5 returns key sets; export them as deterministic JSON lists."""
    return {name: sorted(value) if isinstance(value, set) else value for name, value in info.items()}


def make_trainer_class():
    # Import the GPU stack only after CPU material checks. Pinning matters: _generate is
    # the TRL 1.15.0 eight-field boundary, not a generic compatibility shim.
    import torch
    from trl import GRPOTrainer
    from trl.chat_template_utils import parse_response
    from trl.models.utils import unwrap_model_for_generation

    class G1Trainer(GRPOTrainer):
        def __init__(self, *, g1_rows, g1_root, g1_budgets, g1_cpu_activation_offload=False, **kwargs):
            self.g1_rows = {row["task_id"]: row for row in g1_rows}
            self.g1_root, self.g1_budgets = g1_root, g1_budgets
            self.g1_reward = DeliveryReward(g1_root / "protocol/spec_output_schema.json")
            self.g1_batch_number = 0
            self.g1_cpu_activation_offload = g1_cpu_activation_offload
            super().__init__(**kwargs)

        def _compute_loss(self, model, inputs):
            if self.g1_cpu_activation_offload and torch.is_grad_enabled():
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    return super()._compute_loss(model, inputs)
            return super()._compute_loss(model, inputs)

        def _get_per_token_logps_and_entropies(
            self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kwargs,
        ):
            # Validation also arrives in groups of four. Score probabilities one sequence
            # at a time so it does not require four long-context forward passes in parallel.
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, batch_size=1, **kwargs,
            )

        def _generate(self, prompts):
            mode = "train" if self.model.training else "eval"
            group_size = self.num_generations if mode == "train" else self.num_generations_eval
            if len(prompts) % group_size:
                raise ValueError("G1 candidate batch is not divisible by the GRPO group size")
            self.g1_batch_number += 1
            trajectories, task_ids = [], []
            candidate_root = Path(self.args.output_dir) / "rollouts" / (
                f"{mode}-step{self.state.global_step:04d}-batch{self.g1_batch_number:06d}"
            )
            was_training = self.model.training
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as actor:
                actor.eval()
                try:
                    def generate(context, maximum, deadline):
                        import time
                        ids = torch.tensor([context], dtype=torch.long, device=self.accelerator.device)
                        kwargs = {
                            "input_ids": ids, "attention_mask": torch.ones_like(ids),
                            "generation_config": copy.deepcopy(self.generation_config),
                            "max_new_tokens": maximum,
                            "max_time": max(0.001, deadline - time.monotonic()),
                            "eos_token_id": self._tokenizer.eos_token_id,
                            "pad_token_id": self._tokenizer.pad_token_id,
                            "use_cache": True, "logits_to_keep": 1,
                            "return_dict_in_generate": False,
                        }
                        with torch.no_grad():
                            output = actor.generate(**kwargs)
                        tokens = output[0, ids.shape[1]:].tolist()
                        if self._tokenizer.eos_token_id in tokens:
                            tokens = tokens[:tokens.index(self._tokenizer.eos_token_id) + 1]
                        return tokens

                    for index, prompt in enumerate(prompts):
                        user = json.loads(prompt[-1]["content"].removeprefix("SPEC INPUT\n"))
                        task_id = user["task_id"]
                        row = self.g1_rows[task_id]
                        if prompt != row["prompt"]:
                            raise ValueError("G1 rollout prompt differs from the frozen task")
                        trajectory = run_trajectory(
                            prompt=prompt, repository=repository_for(self.g1_root, row),
                            tokenizer=self._tokenizer, generate=generate,
                            parse_response=lambda sampled, prefix: parse_response(
                                self._tokenizer, sampled, prefix=prefix,
                            ),
                            reward=self.g1_reward, budgets=self.g1_budgets,
                        )
                        trajectories.append(trajectory)
                        task_ids.append(task_id)
                        save_trajectory(
                            candidate_root / f"{index:02d}-{task_id}", trajectory,
                            task_id=task_id, step=self.state.global_step, mode=mode,
                            candidate=index % group_size, group=index // group_size,
                        )
                        print(json.dumps({"experiment": "G1", "mode": mode,
                                          "step": self.state.global_step, "task_id": task_id,
                                          "candidate": index % group_size, **trajectory.score}), flush=True)
                finally:
                    actor.train(was_training)
            rewards = [t.score["reward"] for t in trajectories]
            if any(len(set(task_ids[i:i + group_size])) != 1
                   for i in range(0, len(task_ids), group_size)):
                raise ValueError("GRPO groups must compare candidates for the same task")
            groups = [rewards[i:i + group_size] for i in range(0, len(rewards), group_size)]
            mixed = sum(len(set(group)) > 1 for group in groups)
            generated = [sum(t.env_mask) for t in trajectories]
            if mode == "train":
                self.state.num_input_tokens_seen += sum(len(t.prompt_ids) for t in trajectories) + sum(generated)
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
            self._metrics[mode]["g1/mixed_group_fraction"].append(mixed / len(groups))
            self._metrics[mode]["g1/ledger_delivery_rate"].append(sum(rewards) / len(rewards))
            self._metrics[mode]["completions/mean_length"].append(sum(generated) / len(generated))
            write_json(candidate_root / "batch.json", {
                "step": self.state.global_step, "mode": mode, "tasks": task_ids,
                "rewards": rewards, "mixed_groups": mixed, "groups": len(groups),
                "generation_backend": "current_in_process_actor_sequential",
            })
            # TRL owns relative advantages, old-policy probabilities, KL and optimization.
            # It recomputes probabilities for the HF backend; observations/header tokens have
            # mask zero, and exact sampled assistant token IDs have mask one.
            return (
                [t.prompt_ids for t in trajectories],
                [t.completion_ids for t in trajectories],
                [t.env_mask for t in trajectories],
                [t.messages for t in trajectories],
                None,
                {"g1_reward": rewards, "g1_termination": [t.termination for t in trajectories]},
                None, [],
            )

    return G1Trainer


def load_base(config: dict, model_path: Path | None, protocol: Path):
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    if model_path is not None:
        if (model_path / "adapter_config.json").exists():
            raise ValueError("G1 initialization must be an original base snapshot, not an adapter")
        if model_path.name != config["model_revision"]:
            raise ValueError("Use the original Hugging Face snapshot directory named by the pinned revision")
    source = str(model_path) if model_path is not None else config["model"]
    options = {"local_files_only": True} if model_path else {"revision": config["model_revision"]}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    full_config = AutoConfig.from_pretrained(source, **options)
    text_config = full_config.text_config
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        source, config=text_config, dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."}, **options,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Original text backbone did not load completely: {loading}")
    tokenizer.chat_template = (protocol / "training_chat_template.jinja").read_text()
    tokenizer.response_schema = json.loads((protocol / "response_schema.json").read_text())
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, loading_report(loading)


def run(args) -> None:
    root = args.materials.resolve()
    checked = verify_materials(root)
    # Validate the schema and frozen protocol even for --check-materials.
    DeliveryReward(root / "protocol/spec_output_schema.json")
    config = json.loads((root / "config.json").read_text())
    if config["trainer"]["max_completion_length"] != config["rollout"]["max_trajectory_tokens"]:
        raise ValueError("TRL and G1 trajectory budgets differ")
    if args.check_materials:
        print(json.dumps(checked, ensure_ascii=False))
        return
    if args.output is None:
        raise ValueError("Provide a new --output directory to preserve results")
    if args.mode == "train" and (args.adapter or args.split != "validation"):
        raise ValueError("G1 training starts from base; --adapter/--split are evaluation-only")
    if args.smoke and args.mode != "train":
        raise ValueError("--smoke is a two-step training check")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This G1 runner targets one GPU with sequential rollout sampling")
    if shutil.which("rg") is None:
        raise RuntimeError("Install ripgrep on the training host before G1 tool sampling")
    for package, expected in (("trl", "1.15.0"), ("transformers", "5.5.0")):
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"G1 requires {package}=={expected}")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig

    if not torch.cuda.is_available():
        raise RuntimeError("G1 training/evaluation requires a CUDA GPU")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol").mkdir()
    for file in (root / "protocol").iterdir():
        shutil.copyfile(file, output / "protocol" / file.name)
    shutil.copyfile(root / "data/manifest.json", output / "material_manifest.json")
    report = {
        "experiment": "G1", "status": "loading_base", "mode": args.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "initialization": config["initialization"], "config": config,
        "adapter_for_evaluation": str(args.adapter) if args.adapter else None,
        "cpu_activation_offload": config.get("cpu_activation_offload", False),
        "skip_base_validation": args.skip_base_validation,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "trl", "peft", "datasets", "accelerate", "liger-kernel",
        )},
    }
    write_json(output / "result.json", report)
    try:
        set_seed(config["trainer"]["seed"])
        model, tokenizer, loading = load_base(config, args.model_path, root / "protocol")
        write_json(output / "base_loading.json", loading)
        (output / "protocol/actual_chat_template.jinja").write_text(tokenizer.chat_template)
        module_names = [name for name, layer in model.named_modules()
                        if isinstance(layer, torch.nn.Linear) and name != "lm_head"]
        lora = LoraConfig(task_type="CAUSAL_LM", target_modules=module_names, **config["lora"])
        if args.adapter:
            model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
            lora = None
        train_rows = load_rows(root, "train")
        validation_rows = load_rows(root, "validation")
        rows = train_rows + validation_rows + load_rows(root, "test")
        settings = dict(config["trainer"])
        if args.smoke:
            train_rows = [row for row in train_rows if row["task_id"] == config["smoke"]["task_id"]]
            validation_rows = validation_rows[:1]
            settings.update(max_steps=config["smoke"]["max_steps"], warmup_steps=0,
                            save_steps=1, eval_strategy="no")
        if args.mode == "evaluate":
            validation_rows = load_rows(root, args.split)
            if args.task_id:
                validation_rows = [row for row in validation_rows if row["task_id"] == args.task_id]
            if not validation_rows:
                raise ValueError("No G1 evaluation tasks selected")
        elif args.task_id:
            raise ValueError("--task-id selects evaluation tasks; use --smoke for one-task training")
        trainer_class = make_trainer_class()
        trainer = trainer_class(
            model=model, processing_class=tokenizer, peft_config=lora,
            args=GRPOConfig(output_dir=str(output), **settings),
            reward_funcs=binary_delivery_reward,
            train_dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(validation_rows),
            g1_rows=rows, g1_root=root, g1_budgets=Budgets(**config["rollout"]),
            g1_cpu_activation_offload=config.get("cpu_activation_offload", False),
        )
        metrics = TrainingMetrics(output)

        class MetricsCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    metrics.record(state.global_step, logs)

        trainer.add_callback(MetricsCallback())
        report["trainable_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        watched = next((p for name, p in trainer.model.named_parameters() if "lora_B" in name), None)
        before = watched.detach().float().cpu().clone() if watched is not None else None
        report["selected_train_tasks"] = [row["task_id"] for row in train_rows]
        report["selected_evaluation_tasks"] = [row["task_id"] for row in validation_rows]
        report["status"] = "running"
        write_json(output / "result.json", report)
        if args.mode == "evaluate":
            report["evaluation"] = trainer.evaluate()
        else:
            if not args.smoke and not args.skip_base_validation:
                report["base_validation"] = trainer.evaluate()
                write_json(output / "result.json", report)
            elif not args.smoke:
                report["base_validation"] = {"status": "skipped", "reason": "--skip-base-validation"}
                write_json(output / "result.json", report)
            result = trainer.train()
            trainer.save_state()
            report["training"] = result.metrics
            trainer.save_model(str(output / "final"))
            tokenizer.save_pretrained(output / "final")
            report["final_validation"] = trainer.evaluate()
            report["global_step"] = trainer.state.global_step
            if before is not None:
                report["watched_lora_B_max_change"] = (watched.detach().float().cpu() - before).abs().max().item()
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "result.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, default=HERE)
    parser.add_argument("--check-materials", action="store_true")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--model-path", type=Path, help="Pinned original Hugging Face snapshot directory")
    parser.add_argument("--adapter", type=Path, help="Evaluation only; never initializes G1 training")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--task-id", help="Evaluate one task from the selected split")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-base-validation", action="store_true",
                        help="Start training directly; retain scheduled and final validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
