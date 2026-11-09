from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from a_agent_finetune.g1.materials import load_rows, verify_materials
from a_agent_finetune.g1.metrics import TrainingMetrics
from a_agent_finetune.g1.reward import DeliveryReward, binary_delivery_reward
from a_agent_finetune.g1.rollout import Budgets, run_trajectory
from a_agent_finetune.g1.tools import RepositoryTools
from a_agent_finetune.g1.train import loading_report

ROOT = Path(__file__).resolve().parents[1]
LEDGER = {
    "summary": "Requested behavior", "obligations": [{
        "statement": "Expose the requested public API", "importance": "required",
        "sources": [{"kind": "task", "field": "description", "quote": "requested public API"}],
    }], "preservation_obligations": [], "open_questions": [],
}


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.reward = DeliveryReward()

    def score(self, value, termination="stop"):
        return self.reward.score(json.dumps(value), termination)

    def test_valid_nonempty_ledger(self):
        self.assertEqual(self.score(LEDGER).reward, 1)

    def test_empty_optional_arrays_allowed(self):
        self.assertEqual(self.score(LEDGER).reason, "delivered")

    def test_empty_obligations_rejected(self):
        value = {**LEDGER, "obligations": []}
        self.assertEqual(self.score(value).reason, "empty_obligations")

    def test_whitespace_is_not_nonempty_content(self):
        value = copy.deepcopy(LEDGER)
        value["obligations"][0]["statement"] = " \n\t"
        self.assertEqual(self.score(value).reason, "blank_text")

    def test_fences_and_surrounding_prose_allowed(self):
        for text in ("```json\n" + json.dumps(LEDGER) + "\n```",
                     "Here is the ledger: " + json.dumps(LEDGER) + "\nReview complete."):
            self.assertEqual(self.reward.score(text, "stop").reward, 1)

    def test_complete_json_after_an_unrelated_object(self):
        text = '{"status":"done"}\n' + json.dumps(LEDGER)
        self.assertEqual(self.reward.score(text, "stop").reward, 1)

    def test_braces_and_escaped_quotes_in_text(self):
        value = copy.deepcopy(LEDGER)
        value["summary"] = 'Support {key} and "quoted" values'
        self.assertEqual(self.reward.score("Result:\n" + json.dumps(value), "stop").reward, 1)

    def test_incomplete_json_is_not_repaired(self):
        text = json.dumps(LEDGER)
        self.assertEqual(self.reward.score("```json\n" + text[:-1], "stop").reward, 0)

    def test_invalid_parent_does_not_reward_a_nested_ledger(self):
        self.assertEqual(self.score({"ledger": LEDGER}).reward, 0)

    def test_surrounding_whitespace_allowed(self):
        self.assertEqual(self.reward.score(" \n" + json.dumps(LEDGER) + "\n", "stop").reward, 1)

    def test_required_fields_and_controller_ids(self):
        value = copy.deepcopy(LEDGER)
        del value["summary"]
        self.assertEqual(self.score(value).reason, "invalid_schema")
        value = copy.deepcopy(LEDGER)
        value["obligations"][0]["id"] = "O1"
        self.assertEqual(self.score(value).reward, 0)

    def test_sources_required_structurally(self):
        value = copy.deepcopy(LEDGER)
        value["obligations"][0]["sources"] = []
        self.assertEqual(self.score(value).reason, "invalid_schema")

    def test_normal_termination_required_even_for_valid_json(self):
        for termination in ("timeout", "length", "token_budget", "context_budget", "malformed_response"):
            self.assertEqual(self.score(LEDGER, termination).reward, 0)

    def test_semantics_and_count_not_rewarded(self):
        value = copy.deepcopy(LEDGER)
        value["obligations"][0]["sources"][0]["quote"] = "Intentionally nonexistent quote"
        value["obligations"] *= 20
        self.assertEqual(self.score(value).reward, 1)

    def test_duplicate_keys_rejected_but_complete_objects_can_be_extracted(self):
        text = json.dumps(LEDGER)
        self.assertEqual(self.reward.score(text + text, "stop").reward, 1)
        self.assertEqual(self.reward.score('{"summary":"a",' + text[1:], "stop").reward, 0)

    def test_non_json_constants_rejected(self):
        text = json.dumps(LEDGER).replace('"Requested behavior"', 'NaN')
        self.assertEqual(self.reward.score("Result: " + text, "stop").reason, "invalid_json")

    def test_wrapped_json_does_not_override_failed_termination(self):
        text = "```json\n" + json.dumps(LEDGER) + "\n```"
        for termination in ("timeout", "length", "token_budget", "context_budget"):
            self.assertEqual(self.reward.score(text, termination).reward, 0)

    def test_boolean_and_integer_types_not_interchangeable(self):
        value = copy.deepcopy(LEDGER)
        value["open_questions"] = [{"question": "Which variant?", "blocking": "false",
                                    "sources": value["obligations"][0]["sources"]}]
        self.assertEqual(self.score(value).reward, 0)

    def test_callback_only_accepts_binary_scores_with_matching_length(self):
        self.assertEqual(binary_delivery_reward([[], []], [0.0, 1.0]), [0.0, 1.0])
        with self.assertRaises(ValueError):
            binary_delivery_reward([[]], [0.5])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        (self.root / "sample.txt").write_text("one\ntwo\nthree\n")
        self.tools = RepositoryTools(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_numbered_reads_and_virtual_root(self):
        self.assertEqual(self.tools.read("/work/base/sample.txt", offset=2, limit=1), "2: two")

    def test_symlink_and_parent_escape_blocked(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("private")
        (self.root / "link").symlink_to(outside)
        for path in ("../outside.txt", "link", str(outside)):
            result = self.tools.execute("read", {"path": path})
            self.assertTrue(result.startswith("Tool error:"))
            self.assertNotIn("private", result)

    def test_model_tool_error_is_observation_not_reward(self):
        result = self.tools.execute("read", {"path": "missing.txt"})
        self.assertTrue(result.startswith("Tool error:"))
        self.assertTrue(self.tools.calls[-1]["error"])

    def test_no_write_or_shell_tools(self):
        self.tools.execute("bash", {"command": "touch modified"})
        self.assertFalse((self.root / "modified").exists())

    def test_find_and_grep(self):
        self.assertEqual(self.tools.find("*.txt"), "sample.txt")
        self.assertIn("sample.txt:2:two", self.tools.grep("two", literal=True))


class FakeTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        if messages[0]["content"] == "dummy":
            return [99, 1, 98] if messages[-1]["role"] == "assistant" else [99, 1, 98, 90, 91]
        return [10, 11]

    def decode(self, ids, **kwargs):
        return json.dumps(LEDGER) if ids[0] == 24 else "tool-call"


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "sample.txt").write_text("public contract\n")
        self.reward = DeliveryReward()
        self.prompt = [{"role": "user", "content": "task"}]

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, sequences, budgets=None, parse=None, clock=None):
        samples = iter(sequences)
        contexts = []

        def generate(context, maximum, deadline):
            contexts.append(list(context))
            return next(samples)

        def parser(sampled, prefix):
            if sampled[0] == 23:
                return {"tool_calls": [{"type": "function", "function": {
                    "name": "read", "arguments": {"path": "sample.txt"},
                }}]}
            return {"content": json.dumps(LEDGER)}

        kwargs = {"clock": clock} if clock else {}
        trajectory = run_trajectory(
            prompt=self.prompt, repository=self.repository, tokenizer=FakeTokenizer(),
            generate=generate, parse_response=parse or parser, reward=self.reward,
            budgets=budgets or Budgets(), **kwargs,
        )
        return trajectory, contexts

    def test_final_only_delivery(self):
        trajectory, _ = self.run_case([[24, 1]])
        self.assertEqual(trajectory.score["reward"], 1)
        self.assertEqual(trajectory.env_mask, [1, 1])

    def test_wrapped_final_json_is_delivered_without_changing_raw_text(self):
        raw = "Here is the ledger:\n```json\n" + json.dumps(LEDGER) + "\n```"
        with patch.object(FakeTokenizer, "decode", return_value=raw):
            trajectory, _ = self.run_case([[24, 1]])
        self.assertEqual(trajectory.score["reward"], 1)
        self.assertEqual(trajectory.final_text, raw)
        self.assertEqual(trajectory.completion_ids, [24, 1])

    def test_parsed_ledger_cannot_replace_invalid_raw_final_text(self):
        trajectory, _ = self.run_case([[25, 1]])
        self.assertEqual(trajectory.score["reward"], 0)

    def test_tools_are_masked_and_sampled_tokens_are_preserved(self):
        trajectory, contexts = self.run_case([[23, 1], [24, 1]])
        self.assertEqual(contexts[1], [10, 11, 23, 1, 98, 90, 91])
        self.assertEqual(trajectory.completion_ids, [23, 1, 98, 90, 91, 24, 1])
        self.assertEqual(trajectory.env_mask, [1, 1, 0, 0, 0, 1, 1])
        self.assertEqual(trajectory.score["reward"], 1)
        self.assertEqual(trajectory.tools[0]["output"], "1: public contract")

    def test_truncated_valid_json_does_not_get_reward(self):
        trajectory, _ = self.run_case([[24]])
        self.assertEqual(trajectory.termination, "length")
        self.assertEqual(trajectory.score["reward"], 0)

    def test_context_budget_does_not_truncate_or_repair(self):
        trajectory, _ = self.run_case([[23, 1]], Budgets(max_context_tokens=6))
        self.assertEqual(trajectory.termination, "context_budget")
        self.assertEqual(trajectory.completion_ids, [23, 1])
        self.assertEqual(trajectory.score["reward"], 0)

    def test_timeout_after_complete_json_is_still_failure(self):
        ticks = iter([0, 1, 30, 31])
        trajectory, _ = self.run_case([[24, 1]], Budgets(timeout_sec=20), clock=lambda: next(ticks))
        self.assertEqual(trajectory.termination, "timeout")
        self.assertEqual(trajectory.score["reward"], 0)

    def test_parser_failure_is_a_model_failure(self):
        def parser(*args):
            raise ValueError("Malformed tool call")
        trajectory, _ = self.run_case([[23, 1]], parse=parser)
        self.assertEqual(trajectory.termination, "malformed_response")
        self.assertEqual(trajectory.score["reward"], 0)


class DatasetTests(unittest.TestCase):
    def test_frozen_inventory_and_disjoint_splits(self):
        result = verify_materials(ROOT)
        self.assertEqual(result["counts"], {"train": 13, "validation": 3, "test": 14})

    def test_only_task_and_system_messages(self):
        for split in ("train", "validation", "test"):
            for row in load_rows(ROOT, split):
                self.assertEqual([m["role"] for m in row["prompt"]], ["system", "user"])
                user = json.loads(row["prompt"][1]["content"].removeprefix("SPEC INPUT\n"))
                self.assertEqual(set(user), {"task_id", "base_root", "task", "managed_delivery"})


class MetricsTests(unittest.TestCase):
    def test_loading_diagnostics_can_be_saved_as_json_without_losing_keys(self):
        info = {"missing_keys": set(), "unexpected_keys": {"visual.b", "visual.a"},
                "mismatched_keys": set(), "error_msgs": []}
        result = json.loads(json.dumps(loading_report(info)))
        self.assertEqual(result["unexpected_keys"], ["visual.a", "visual.b"])
        self.assertEqual(result["missing_keys"], [])
        self.assertIsInstance(info["unexpected_keys"], set)

    def test_eval_metrics_preserved_without_creating_training_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = TrainingMetrics(root)
            metrics.record(0, {"eval_loss": 0.1, "eval_reward": 0.5})
            record = json.loads((root / "metrics.jsonl").read_text())
            self.assertEqual(record["eval_reward"], 0.5)
            self.assertFalse((root / "loss.csv").exists())

    def test_loss_and_reward_history_saved_across_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = TrainingMetrics(root)
            with patch.object(metrics, "plot") as plot:
                metrics.record(1, {"loss": -0.02, "g1/ledger_delivery_rate": 0.25})
                metrics.record(2, {"loss": 0.01, "g1/ledger_delivery_rate": 0.75})
            self.assertEqual(plot.call_count, 2)
            self.assertEqual(len((root / "metrics.jsonl").read_text().splitlines()), 2)
            self.assertEqual([row["loss"] for row in metrics.rows], [-0.02, 0.01])
            self.assertIn("0.75", (root / "loss.csv").read_text())


if __name__ == "__main__":
    unittest.main()
