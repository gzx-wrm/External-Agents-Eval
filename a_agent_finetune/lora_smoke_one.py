"""A small BF16 LoRA capacity test using a pretokenized complete A trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--diagnostic-prefix-tokens", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting", "model": "Qwen/Qwen3.5-9B",
        "revision": args.model_path.name, "task_id": "termenv-preserve-ansi-resets",
        "full_trajectory": args.diagnostic_prefix_tokens == 0,
        "diagnostic_prefix_tokens": args.diagnostic_prefix_tokens,
        "precision": "bf16", "quantized": False, "text_only": True, "rank": 16, "alpha": 32,
        "batch_size": 1, "requested_steps": args.steps, "completed_steps": [],
        "start_time": time.time(),
    }

    def event(stage: str, **values) -> None:
        row = {"time": time.time(), "stage": stage, **values}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["stage"] = stage
        write_json(args.output / "result.json", report)

    try:
        # Unsloth must patch Transformers before PEFT/Transformers are imported.
        from unsloth import FastLanguageModel
        import numpy as np
        import torch
        from bitsandbytes.optim import AdamW8bit
        from transformers import AutoTokenizer

        torch.set_num_threads(8)
        torch.manual_seed(3407)
        report["versions"] = {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "unsloth", "unsloth_zoo", "peft", "trl",
            "bitsandbytes", "flash-linear-attention",
        )}
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_capacity_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        sample = json.loads((args.materials / "sample.json").read_text())
        template = (args.materials / "protocol/training_chat_template.jinja").read_text()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
        encoded = tokenizer.apply_chat_template(
            sample["messages"], tools=sample["tools"], chat_template=template,
            tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
            enable_thinking=True, add_generation_prompt=False,
        )
        with np.load(args.materials / "sample.npz") as arrays:
            original_ids = arrays["input_ids"].copy()
            original_labels = arrays["labels"].copy()
        if not np.array_equal(encoded["input_ids"], original_ids):
            raise ValueError("Transformers reserialization differs from frozen input_ids")
        if not np.array_equal(encoded["assistant_masks"], original_labels != -100):
            raise ValueError("Transformers assistant mask differs from frozen loss mask")
        report["tokenizer_and_loss_mask_verified"] = True
        report["original_tokens"] = len(original_ids)
        report["original_supervised_tokens"] = int(np.count_nonzero(original_labels != -100))
        if args.diagnostic_prefix_tokens:
            original_ids = original_ids[:args.diagnostic_prefix_tokens]
            original_labels = original_labels[:args.diagnostic_prefix_tokens]
        report["tokens_used"] = len(original_ids)
        report["supervised_tokens_used"] = int(np.count_nonzero(original_labels != -100))
        if report["supervised_tokens_used"] == 0:
            raise ValueError("Diagnostic prefix has no supervised assistant tokens")
        event("loading_model", tokens=report["tokens_used"], supervised=report["supervised_tokens_used"])
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.model_path), max_seq_length=len(original_ids),
            dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False,
            load_in_16bit=True, full_finetuning=False, unsloth_tiled_mlp=True,
            text_only=True,
        )
        # Match text linear layers dynamically, including GatedDeltaNet projections.
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and not any(part in name for part in ("visual", "vision", "lm_head", "mtp"))]
        if not targets:
            raise ValueError("No text linear modules were found for LoRA")
        write_json(args.output / "target_modules.json", {"modules": targets})
        model = FastLanguageModel.get_peft_model(
            model, r=16, target_modules="(?:" + "|".join(re.escape(name) for name in targets) + ")",
            lora_alpha=32, lora_dropout=0,
            bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
        )
        FastLanguageModel.for_training(model, use_gradient_checkpointing="unsloth")
        model.config.use_cache = False
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if any("lora_" not in name for name, _ in trainable):
            raise ValueError("Unexpected trainable base-model parameter")
        report["trainable_parameters"] = sum(parameter.numel() for _, parameter in trainable)
        report["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        watched_name, watched = next((name, parameter) for name, parameter in trainable if "lora_B" in name)
        before = watched.detach().float().cpu().clone()
        optimizer = AdamW8bit([parameter for _, parameter in trainable], lr=1e-4, weight_decay=0)
        input_ids = torch.as_tensor(original_ids.astype(np.int64), device="cuda").unsqueeze(0)
        labels = torch.as_tensor(original_labels.astype(np.int64), device="cuda").unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        event("model_ready", trainable_parameters=report["trainable_parameters"],
              allocated_gib=torch.cuda.memory_allocated() / 2**30)
        report["status"] = "training"
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            event("forward_start", step=step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, attention_mask=attention_mask,
                               labels=labels, use_cache=False, return_dict=True)
                loss = output.loss
            loss_value = float(loss.detach())
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at step {step}: {loss_value}")
            event("backward_start", step=step, loss=loss_value)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], 1.0, error_if_nonfinite=True,
            ))
            optimizer.step()
            torch.cuda.synchronize()
            row = {
                "step": step, "loss": loss_value, "grad_norm": grad_norm,
                "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            report["completed_steps"].append(row)
            event("optimizer_step_complete", **row)
            del output, loss
        optimizer.zero_grad(set_to_none=True)
        changed = float((watched.detach().float().cpu() - before).abs().max())
        report["watched_adapter_parameter"] = watched_name
        report["maximum_adapter_parameter_change"] = changed
        if changed == 0:
            raise ValueError("LoRA parameter did not change")
        adapter = args.output / "adapter"
        event("saving_adapter")
        model.save_pretrained(str(adapter), safe_serialization=True)
        tokenizer.chat_template = template
        tokenizer.save_pretrained(str(adapter))
        report["adapter_files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in adapter.iterdir() if path.is_file()
        }
        # Free training tensors before checking that saved adapter weights can load.
        del optimizer, input_ids, labels, attention_mask
        gc.collect()
        torch.cuda.empty_cache()
        event("reloading_adapter")
        model.load_adapter(str(adapter), adapter_name="reload_smoke", is_trainable=False)
        model.set_adapter("reload_smoke")
        loaded = dict(model.named_parameters())[watched_name.replace(".default.", ".reload_smoke.")]
        if not torch.equal(loaded.detach().cpu(), watched.detach().cpu()):
            raise ValueError("Saved/reloaded adapter parameter differs")
        report["adapter_reload_verified"] = True
        report["status"] = "completed"
        event("completed", completed_steps=len(report["completed_steps"]), parameter_change=changed)
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        try:
            import torch
            report["peak_allocated_gib_at_failure"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_gib_at_failure"] = torch.cuda.max_memory_reserved() / 2**30
        except Exception:
            pass
        event("failed", error_type=report["error_type"], error=report["error"])
        raise
    finally:
        report["end_time"] = time.time()
        write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
