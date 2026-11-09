from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/deepswe_ft_m1/ledger_input.py"
if not PATH.exists():
    PATH = Path(__file__).resolve().parents[1] / "scripts/ledger_input.py"
spec = importlib.util.spec_from_file_location("ft_m1_ledger_input", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenLedgerInputTests(unittest.TestCase):
    def setUp(self):
        self.ledger = {
            "summary": "Preserve existing behavior while adding the requested feature.",
            "obligations": [{"id": "O001", "content": {
                "statement": "Implement the requested behavior.",
                "sources": [{"kind": "repository", "path": "src/public.py", "line_start": 2, "line_end": 4}],
            }}],
            "preservation_obligations": [{"id": "P001", "content": {"statement": "Keep the default behavior."}}],
            "open_questions": [{"id": "Q001", "content": {"question": "Which optional default applies?", "blocking": False}}],
        }

    def test_only_instruction_is_extended_and_ledger_is_complete(self):
        task = {"task_id": "example", "instruction": "Original issue with trailing space. \n",
                "task_checksum": "unchanged", "image": "original"}
        original = dict(task)
        output = module.augment_task(task, self.ledger)
        self.assertEqual(task, original)
        self.assertEqual({k: v for k, v in output.items() if k != "instruction"},
                         {k: v for k, v in task.items() if k != "instruction"})
        self.assertTrue(output["instruction"].startswith(task["instruction"]))
        block = output["instruction"].split("```json\n", 1)[1].rsplit("\n```", 1)[0]
        self.assertEqual(json.loads(block), self.ledger)
        self.assertEqual(output["instruction"].count(module.GUIDANCE), 1)
        self.assertEqual(output["instruction"].count(module.HEADING), 1)

    def test_modified_frozen_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = json.dumps(self.ledger).encode()
            (root / "ledger.json").write_bytes(data)
            record = {"task_id": "example", "ledger_file": "ledger.json",
                      "ledger_file_sha256": hashlib.sha256(data).hexdigest(),
                      "ledger_sha256": module.canonical_sha256(self.ledger)}
            self.assertEqual(module.load_frozen_ledger(root, record), self.ledger)
            (root / "ledger.json").write_bytes(data + b" ")
            with self.assertRaisesRegex(ValueError, "Frozen ledger file changed"):
                module.load_frozen_ledger(root, record)


if __name__ == "__main__":
    unittest.main()
