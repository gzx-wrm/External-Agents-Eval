from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/deepswe_ft_m1_gpt/prepare_remaining.py"
SPEC = importlib.util.spec_from_file_location("gpt_remaining_selection", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class RemainingSelectionTests(unittest.TestCase):
    def test_complement_preserves_order_without_repeating_previous_cases(self):
        suite = ["task-" + str(i) for i in range(30)]
        previous = [suite[i] for i in [21, 4, 16, 0, 29]]
        remaining = module.remaining_ids(suite, previous)
        self.assertEqual(len(remaining), 25)
        self.assertEqual(remaining, [x for x in suite if x not in previous])
        self.assertFalse(set(remaining) & set(previous))
        self.assertEqual(set(remaining) | set(previous), set(suite))

    def test_unknown_or_duplicate_previous_cases_are_rejected(self):
        suite = ["task-" + str(i) for i in range(30)]
        for previous in [suite[:4] + ["unknown"], suite[:4] + [suite[0]]]:
            with self.assertRaises(ValueError):
                module.remaining_ids(suite, previous)


if __name__ == "__main__":
    unittest.main()
