"""BF16 LoRA training on frozen, complete A-Agent trajectories (batch size one)."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
import traceback
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def learning_rate_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["batch_size"] != 1 or config["gradient_accumulation_steps"] != 1:
        raise ValueError("This full-trajectory runner uses batch size one without accumulation")
    if config["packing"] or config["truncation"]:
        raise ValueError("This runner preserves each complete trajectory")
    manifest_path = args.materials / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    total_steps = len(tasks) * config["epochs"]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting", "model": manifest["model"],
        "revision": manifest["model_revision"], "config": config,
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "full_trajectories": True, "precision": "bf16", "quantized": False,
        "text_only": True, "requested_steps": total_steps,
        "completed_steps": [], "checkpoints": [], "start_time": time.time(),
    }

    def event(stage: str, **values) -> None:
        row = {"time": time.time(), "stage": stage, **values}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["stage"] = stage
        write_json(args.output / "result.json", report)

    try:
        # Patch Transformers before importing it or PEFT.
        from unsloth import FastLanguageModel
        import numpy as np
        import torch
        from bitsandbytes.optim import AdamW8bit
        from transformers import AutoTokenizer

        torch.set_num_threads(8)
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        report["versions"] = {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "unsloth", "unsloth_zoo", "peft", "trl",
            "bitsandbytes", "flash-linear-attention",
        )}
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_capacity_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        if args.model_path.name != manifest["model_revision"]:
            raise ValueError("Base-model revision differs from the frozen dataset")
        for name, expected in manifest["file_sha256"].items():
            if hashlib.sha256((args.materials / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Dataset file hash differs: {name}")

        template = (args.materials / "protocol/training_chat_template.jinja").read_text()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
        lines = (args.materials / "train.jsonl").read_text().splitlines()
        if len(lines) != len(tasks):
            raise ValueError("Training rows differ from the manifest")
        samples = []
        for task, line in zip(tasks, lines):
            sample = json.loads(line)
            encoded = tokenizer.apply_chat_template(
                sample["messages"], tools=sample["tools"], chat_template=template,
                tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                enable_thinking=True, add_generation_prompt=False,
            )
            with np.load(args.materials / "tokenized" / f"{task['task_id']}.npz") as arrays:
                ids, labels = arrays["input_ids"].copy(), arrays["labels"].copy()
            if not np.array_equal(encoded["input_ids"], ids):
                raise ValueError(f"Frozen input_ids differ: {task['task_id']}")
            if not np.array_equal(encoded["assistant_masks"], labels != -100):
                raise ValueError(f"Frozen assistant loss mask differs: {task['task_id']}")
            if len(ids) != task["total_tokens"] or int(np.count_nonzero(labels != -100)) != task["supervised_tokens"]:
                raise ValueError(f"Token counts differ: {task['task_id']}")
            if len(ids) > config["max_seq_length"]:
                raise ValueError(f"Full trajectory exceeds context limit: {task['task_id']}")
            samples.append((ids, labels))
            event("sample_verified", task_id=task["task_id"], tokens=len(ids),
                  supervised_tokens=task["supervised_tokens"])
        del lines, sample, encoded
        report["tokenizer_and_loss_masks_verified"] = True
        report["sample_count"] = len(samples)
        report["tokens_per_epoch"] = sum(len(ids) for ids, _ in samples)
        report["supervised_tokens_per_epoch"] = sum(task["supervised_tokens"] for task in tasks)
        event("loading_model")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.model_path), max_seq_length=config["max_seq_length"],
            dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False,
            load_in_16bit=True, full_finetuning=False, unsloth_tiled_mlp=True,
            text_only=True,
        )
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and not any(part in name for part in ("visual", "vision", "lm_head", "mtp"))]
        if not targets:
            raise ValueError("No text linear modules were found for LoRA")
        write_json(args.output / "target_modules.json", {"modules": targets})
        model = FastLanguageModel.get_peft_model(
            model, r=config["lora_rank"],
            target_modules="(?:" + "|".join(re.escape(name) for name in targets) + ")",
            lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            bias="none", use_gradient_checkpointing="unsloth", random_state=seed,
        )
        FastLanguageModel.for_training(model, use_gradient_checkpointing="unsloth")
        model.config.use_cache = False
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if any("lora_" not in name for name, _ in trainable):
            raise ValueError("Unexpected trainable base-model parameter")
        parameters = [parameter for _, parameter in trainable]
        report["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
        report["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        watched_name, watched = next((name, parameter) for name, parameter in trainable if "lora_B" in name)
        before = watched.detach().float().cpu().clone()
        optimizer = AdamW8bit(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_factor(
                step, warmup_steps=config["warmup_steps"], total_steps=total_steps),
        )
        tokenizer.chat_template = template
        event("model_ready", trainable_parameters=report["trainable_parameters"])
        report["status"] = "training"
        global_step = 0
        for epoch in range(1, config["epochs"] + 1):
            order = list(range(len(tasks)))
            rng.shuffle(order)
            epoch_rows = []
            event("epoch_start", epoch=epoch, task_order=[tasks[index]["task_id"] for index in order])
            for index in order:
                task = tasks[index]
                ids, label_values = samples[index]
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                input_ids = torch.as_tensor(ids.astype(np.int64), device="cuda").unsqueeze(0)
                labels = torch.as_tensor(label_values.astype(np.int64), device="cuda").unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                lr = optimizer.param_groups[0]["lr"]
                event("forward_start", epoch=epoch, step=global_step,
                      task_id=task["task_id"], tokens=len(ids), learning_rate=lr)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels, use_cache=False, return_dict=True)
                    loss = output.loss
                loss_value = float(loss.detach())
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {global_step}: {loss_value}")
                event("backward_start", epoch=epoch, step=global_step,
                      task_id=task["task_id"], loss=loss_value)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    parameters, config["max_grad_norm"], error_if_nonfinite=True,
                ))
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize()
                row = {
                    "epoch": epoch, "step": global_step, "task_id": task["task_id"],
                    "label_source": task["label_source"], "tokens": len(ids),
                    "supervised_tokens": task["supervised_tokens"],
                    "learning_rate": lr, "loss": loss_value, "grad_norm": grad_norm,
                    "seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
                report["completed_steps"].append(row)
                epoch_rows.append(row)
                event("optimizer_step_complete", **row)
                del output, loss, input_ids, labels, attention_mask
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint = args.output / f"checkpoint-epoch-{epoch}"
            checkpoint.mkdir()
            adapter = checkpoint / "adapter"
            event("saving_checkpoint", epoch=epoch, step=global_step)
            model.save_pretrained(str(adapter), safe_serialization=True)
            tokenizer.save_pretrained(str(adapter))
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "shuffle_rng_state": rng.getstate(), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(), "config": config,
                "dataset_manifest_sha256": report["dataset_manifest_sha256"],
            }, checkpoint / "training-state.pt")
            checkpoint_row = {
                "epoch": epoch, "step": global_step, "path": checkpoint.name,
                "mean_task_loss": sum(row["loss"] for row in epoch_rows) / len(epoch_rows),
                "adapter_files": {
                    path.name: {"bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in adapter.iterdir() if path.is_file()
                },
            }
            write_json(checkpoint / "summary.json", checkpoint_row)
            report["checkpoints"].append(checkpoint_row)
            event("epoch_complete", epoch=epoch, step=global_step,
                  mean_task_loss=checkpoint_row["mean_task_loss"])
        changed = float((watched.detach().float().cpu() - before).abs().max())
        report["maximum_adapter_parameter_change"] = changed
        if changed == 0:
            raise ValueError("LoRA parameter did not change")
        del optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        event("reloading_final_adapter")
        model.load_adapter(str(adapter), adapter_name="reload_check", is_trainable=False)
        model.set_adapter("reload_check")
        loaded = dict(model.named_parameters())[watched_name.replace(".default.", ".reload_check.")]
        if not torch.equal(loaded.detach().cpu(), watched.detach().cpu()):
            raise ValueError("Saved/reloaded adapter parameter differs")
        report["adapter_reload_verified"] = True
        report["status"] = "completed"
        event("completed", completed_steps=global_step, checkpoints=len(report["checkpoints"]))
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        event("failed", error_type=report["error_type"], error=report["error"])
        raise
    finally:
        report["end_time"] = time.time()
        write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
