from __future__ import annotations

from typing import Any

from external_agents_eval.agents.pier.pi_v6_solve_review_repair import (
    PiV6SolveReviewRepairPierAgent,
)
from external_agents_eval.workflows.solve_review_repair.ft_inference import FTLedgerClient
from external_agents_eval.workflows.solve_review_repair.v62_config import V62Config
from external_agents_eval.workflows.solve_review_repair.v62_controller import V62Controller


class PiV62SolveReviewRepairPierAgent(PiV6SolveReviewRepairPierAgent):
    """Run the unchanged FT protocol between A exploration and V6.1 Solver."""

    workflow_version = "6.2"
    workflow_config_type = V62Config

    @staticmethod
    def name() -> str:
        return "pi-v62-review-repair"

    def _make_controller(self, **kwargs: Any) -> V62Controller:
        model = self.model_spec_for_role("A")
        return V62Controller(
            ledger_client=FTLedgerClient(
                model=model.model, base_url=model.effective_base_url,
                api_key=self.extra_env[model.api_key_env],
            ),
            **kwargs,
        )
