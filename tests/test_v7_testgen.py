from external_agents_eval.workflows.acceptance_gate import v7_testgen as tg

INSTRUCTION = """Add a `retry` option.

`initialize()` is idempotent, calling it multiple times after success returns immediately.
"""


def new_file(path, body="x"):
    return (f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..1111111\n"
            f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+{body}\n")


def modified(path):
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
            "@@ -1 +1 @@\n-a\n+b\n")


def test_tg_test_patch_keeps_only_new_test_files_and_drops_the_baseline():
    env = modified("package-lock.json")
    snapshot = env + new_file("src/foo.test.ts") + new_file("src/impl.ts") + modified("src/a.ts")
    patch, files, violations = tg.tg_test_patch(snapshot, env)
    assert files == ["src/foo.test.ts"]
    assert "src/impl.ts" not in patch and "package-lock.json" not in patch
    assert {v["path"]: v["reason"] for v in violations} == {
        "src/impl.ts": "new non-test file", "src/a.ts": "modified existing file"}


def test_quote_must_be_contiguous_task_text():
    assert tg.quote_in_task("calling it multiple   times after success", INSTRUCTION)
    assert not tg.quote_in_task("calling it twice after success", INSTRUCTION)
    assert not tg.quote_in_task("retry", INSTRUCTION)  # too short to bind a requirement


def manifest(**test):
    base = {"id": "T1", "requirements": ["R1"], "kind": "new", "files": ["tests/test_x.py"],
            "command": "pytest tests/test_x.py -q"}
    return {"requirements": [{"id": "R1", "quote": "is idempotent, calling it multiple times"}],
            "tests": [{**base, **test}]}


def test_validate_manifest_binds_quotes_and_files():
    checked = tg.validate_manifest(manifest(), INSTRUCTION, ["tests/test_x.py"])
    assert checked["tests"][0]["valid"] and checked["tests"][0]["quote_ok"]
    missing = tg.validate_manifest(manifest(), INSTRUCTION, [])
    assert not missing["tests"][0]["valid"]
    existing = tg.validate_manifest(manifest(kind="existing", files=[], requirements=[]), INSTRUCTION, [])
    assert existing["tests"][0]["valid"]
    assert tg.validate_manifest(None, INSTRUCTION, [])["problems"]


def test_classification_follows_kind_and_baseline():
    t = tg.validate_manifest(manifest(), INSTRUCTION, ["tests/test_x.py"])["tests"][0]
    assert tg.classify(t, {"returncode": 1}) == "hard"
    assert tg.classify(t, {"returncode": 0}) == "base_pass"
    keep = {**t, "kind": "preserve"}
    assert tg.classify(keep, {"returncode": 0}) == "hard"
    assert tg.classify(keep, {"returncode": 1}) == "base_fail"
    assert tg.classify({**t, "quote_ok": False}, {"returncode": 1}) == "unquoted"


def test_candidate_verdict_requires_two_failures():
    t = tg.validate_manifest(manifest(), INSTRUCTION, ["tests/test_x.py"])["tests"][0]
    base = {"T1": {"returncode": 1}}
    assert tg.candidate_verdict([t], base, {"T1": {"returncode": 0}}, {})["reject"] is False
    flaky = tg.candidate_verdict([t], base, {"T1": {"returncode": 1}}, {"T1": {"returncode": 0}})
    assert flaky["reject"] is False and flaky["flaky"] == ["T1"]
    assert tg.candidate_verdict([t], base, {"T1": {"returncode": 1}}, {"T1": {"returncode": 1}})["reject"]
    assert not tg.candidate_verdict([t], base, {"T1": {"returncode": 1}}, {"T1": {"returncode": 1}},
                                    exclude={"T1"})["reject"]


def test_tg_config_replaces_only_the_task_section():
    config = {"agent": {"instance_template": "Please solve this issue: {{task}}\n## Command Execution Rules\nrules",
                        "step_limit": 0}, "environment": {"run_args": ["--rm"]}, "model": {}}
    result = tg.tg_config(config, step_limit=10, wall_time_sec=60, command_timeout_sec=30, labels=["--label=x"])
    assert result["agent"]["instance_template"].startswith("You are the V7 acceptance-test author")
    assert result["agent"]["instance_template"].endswith("## Command Execution Rules\nrules")
    assert "{{task}}" in result["agent"]["instance_template"]
    assert result["environment"]["timeout"] == 30 and result["environment"]["run_args"] == ["--rm", "--label=x"]
    assert config["environment"]["run_args"] == ["--rm"]


def test_patch_new_files_reconstructs_contents():
    patch = new_file("tests/test_a.py", "assert True") + modified("src/a.py")
    assert tg.patch_new_files(patch) == {"tests/test_a.py": "assert True\n"}


def test_audit_messages_and_parsing_drop_unsure_or_missing_tests():
    checked = tg.validate_manifest(manifest(), INSTRUCTION, ["tests/test_x.py"])
    messages = tg.audit_messages(INSTRUCTION, checked, new_file("tests/test_x.py", "def test(): pass"))
    assert messages[0]["role"] == "system" and "def test(): pass" in messages[1]["content"]
    decisions, problems = tg.parse_audit('```json\n{"tests": [{"id": "T1", "decision": "keep", "criteria": []}]}\n```',
                                         ["T1", "T2"])
    assert decisions["T1"]["decision"] == "keep" and decisions["T2"]["decision"] == "drop"
    assert problems == ["T2 missing from audit"]
    garbled, problems = tg.parse_audit("not json", ["T1"])
    assert garbled["T1"]["decision"] == "drop" and problems


def test_guessed_names_come_from_baseline_errors_missing_in_task():
    out = ("x_test.go:3: undefined: rego.EvalRuleProfile\nx_test.go:4: undefined: rego.EnableRuleProfile\n"
           "TypeError: client.retry is not a function\nImportError: cannot import name 'Retry'")
    task = "Add `rego.EvalRuleProfile(bool)` and a `retry` option."
    assert tg.missing_names(out) == ["EvalRuleProfile", "EnableRuleProfile", "retry", "Retry"]
    assert tg.guessed_names(out, task) == ["EnableRuleProfile", "Retry"]
