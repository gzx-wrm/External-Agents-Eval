"""Persist Trainer metrics and an updated GRPO training-loss plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class TrainingMetrics:
    columns = (
        "step", "loss", "learning_rate", "grad_norm", "reward", "reward_std", "kl",
        "g1/ledger_delivery_rate", "g1/mixed_group_fraction",
    )

    def __init__(self, output: Path):
        self.output = output
        self.rows: list[dict] = []

    def record(self, step: int, logs: dict) -> None:
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({**logs, "step": step}) + "\n")
        if "loss" not in logs:
            return
        self.rows.append({name: step if name == "step" else logs.get(name, "")
                          for name in self.columns})
        with (self.output / "loss.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)
        self.plot()

    def plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot([row["step"] for row in self.rows], [row["loss"] for row in self.rows],
                  marker="o", markersize=3, linewidth=1.5)
        axis.set(xlabel="Optimizer step", ylabel="GRPO training loss", title="G1 training loss")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(self.output / "loss_curve.png", dpi=160)
        plt.close(figure)
