# CodeScout preprocessing reproduction

This is an independent reimplementation of **CodeScout: Contextual Problem
Statement Enhancement for Software Agents** (Suri et al., Findings ACL 2026).

- Paper: https://aclanthology.org/2026.findings-acl.2032.pdf
- Prompt source: https://arxiv.org/abs/2603.05744v2, Appendix A, CC BY 4.0.
- Appendix B gives the algorithm; Appendix C supplies intermediate/output examples.
- No runnable official implementation or complete output archive was found during
  the 2026-10-29 inspection. This implementation does not claim byte-identical
  reproduction of undisclosed indexing/configuration details.

## Method

1. Parse a frozen repository into files, entities and structural/dependency edges.
2. Apply Appendix A.1 to the original issue and a repository outline.
3. Resolve each selected file/class/function against the index; retrieve frozen
   source bytes, with exact paths, ranges and hashes.
4. Apply Appendix A.2 independently to every successfully retrieved target.
5. Retain analyses whose relevance is at least the configured threshold.
6. Apply Appendix A.3 to the original issue and retained analyses.

Only the synthesized problem statement is intended for native Mini. No FT ledger,
teacher evidence, old solution, review advice or verifier feedback is an input.
The per-target analyses are independent requests, not a shared conversation. There
are no model tools, exploration loop, implementation stage or reviewer in this
preprocessor. A normal instance uses `2 + successfully retrieved targets` requests.

The three prompts preserve Appendix A's wording, headings, fields and requested
5–10 targets; typography/line wrapping is normalized. The controller accepts up
to the main text's hard cap of 15. JSON equivalents of the published text formats
are also parsed. Analysis field headings accept either spaces or underscores;
this handles observed presentation variation without changing their semantics.
Exact duplicate target records are analyzed only once.

## Explicit reproduction choices

`configs/codescout-deepseek-v4.1-flash.json` contains all runtime settings:

| Setting | Initial value and reason |
| --- | --- |
| Relevance threshold | 4/10. The paper does not specify the actual cutoff; its analysis prompt marks scores below 4 as low relevance. This is our choice, not an author-confirmed default. |
| Parser | Tree-sitter 0.25.2 with separately pinned Python, JS, TS/TSX, Go and Rust grammars; installed wheels need no runtime grammar downloads. 0.26.0 was rejected after native crashes and incorrect locations on real repositories in the local Python 3.13/macOS environment. Ranges are independently derived from frozen byte offsets. |
| Graph | JSON nodes/edges with locations, signatures, containment, import declarations and applicable inheritance/implementation relations. Import edges retain declared dependencies; this is not whole-program name resolution or a complete call graph. |
| Scope representation | Complete eligible file list, followed by class/function signatures in deterministic path order. Maximum 240,000 characters. Omitted entity annotations are counted; an oversized file list fails rather than being silently truncated. |
| Source limits | UTF-8 files up to 4,000,000 bytes are indexed. A retrieval has 240,000 characters, shared across ambiguous matches with room for headers. Oversized targets are explicitly unavailable, not silently clipped. |
| Ambiguous names | Prefer qualified names, then public-name suffix matches. Supply all matches when there are at most 8. Larger ambiguities are recorded and skipped. |
| Exclusions | Git metadata, symlinks, dependency/environment/cache directories and Rust build output. Exact skipped paths and reasons are recorded; other unsupported-language text files remain retrievable as files. |
| Parse errors | Keep the file, available AST entities and a parse-error flag. Never report such a file as fully parsed. |
| Failed retrieval | Record and skip the target, as in Algorithm 1. If no target is analyzed, mark preprocessing failed instead of treating a context-free rewrite as valid augmentation. |
| Model | DeepSeek V4.1 Flash alias `deepseek-flash`, high, thinking enabled, max output 65,536; matches the existing native comparison's model settings. |
| Time/retries | 600-second socket timeout/request, 3,600-second budget checked before new calls. No automatic transport or format resampling. Requests without recorded responses need an explicit new run to retry. |

The global time budget is checked between requests; it is not an external process
kill timer. HTTP keepalive traffic can extend a response beyond a socket timeout.
Host-side process supervision can enforce a strict wall limit for a future batch.

## Standalone use

Install the optional parser dependencies in the chosen environment:

```bash
python -m pip install -e '.[codescout]'
python -m external_agents_eval.preprocessors.codescout index \
  --repository /path/to/frozen/repository --output /path/to/separate/index-output
python -m external_agents_eval.preprocessors.codescout run \
  --repository /path/to/frozen/repository --issue /path/to/original-issue.md \
  --config configs/codescout-deepseek-v4.1-flash.json \
  --output /path/to/separate/preprocessing-output
```

`run` reads `DEEPSEEK_API_KEY` from the host process environment. Headers are not
written to artifacts. Every stage is a tool-free, independent chat completion;
the Mini model interface is deliberately not used because it injects Bash tools.
The caller must supply an isolated public baseline, not a directory containing
solutions, hidden tests or historical agent outputs.

## Fixed DeepSWE30 adapter

`run.py` creates a fresh container from the exact M0 image pinned in an existing
FT_M1 bundle. It reuses the byte-identical native repository preparation/sanitizer:
network disabled, synthetic baseline Git, no history, mounts or forwarded keys.
It exports only eligible regular repository files; the host constructs the AST
index without importing or executing project code. The temporary container is
removed. Existing experiment files and Docker images remain intact.

```bash
python scripts/deepswe_codescout/run.py index \
  --native-root /root/ft_m1-deepswe-m0failed8-20261028-w3-r1 \
  --task-id vulture-persistent-analysis-cache \
  --output /root/codescout-new-run/vulture
```

Use `run` instead of `index` to make real preprocessing model calls. Choose the
complementary `m0passed22` bundle for tasks pinned there. `public-tasks.json`
supplies the **original** instruction; FT_M1's augmented input files and ledgers
are never read. The original task checksum and image ID must match the selection.

The output `augmented-task.json` replaces the instruction and pins the image
digest. It is ready for a native-Mini input adapter; this command does not launch
Solver or evaluation and is not a 30-task batch launcher.

## End-to-end batch runner

`batch.py` adds scheduling without modifying CodeScout or the frozen native
Mini/scorer functions. Each of three slots performs preprocessing, fresh-baseline
Solver generation, then immediate official evaluation. Preprocessing has its own
3,600-second subprocess limit; Solver retains its independent 10,800-second
budget and scoring retains the official 1,800 seconds.

```bash
python scripts/deepswe_codescout/batch.py prepare --root /root/codescout-new-batch \
  --source /root/ft_m1-deepswe-m0failed8-20261028-w3-r1 \
  --source /root/ft_m1-deepswe-m0passed22-20261028-w3-r1 \
  --reuse-preprocessing /path/to/codescout-preprocessor-smoke-r2
python scripts/deepswe_codescout/batch.py preflight --root /root/codescout-new-batch
# Supply the credential on stdin, not a command-line argument or saved file.
python scripts/deepswe_codescout/launch.py /root/codescout-new-batch
```

Add `--task-id vulture-persistent-analysis-cache` to `prepare` for the one-case
end-to-end smoke. Without `--task-id`, preparation selects exactly the canonical
30 tasks, including True Myth and excluding SQLite. Both source bundles must
cover those 30 with identical original tasks, native code and configuration.

Preflight uses one available enhanced issue in the real native protocol with
simulated model responses, verifies exact input injection, and references the
unchanged M0 scorer's successful gold/control checks. It makes zero new model
calls when a complete frozen preprocessing cache is supplied. If there is no
cached case, explicitly run `batch.py preprocess --root ... --task-id ...` first.

Reusable preprocessing must match original issue, task checksum, M0 image,
configuration and parser versions. It is verified against a new isolated source
snapshot; an incomplete imported cache cannot make new model calls. The new
Solver always starts from a separate original-baseline container.

`monitor-status.json` records each phase; each case has `preprocessing/`,
`generation/`, `evaluation/` and `result.json` under `outputs/<task>/`. Native Mini
messages, Bash protocol and model compatibility code are byte-for-byte M0 copies.
Usage separates preprocessing and Solver, and records cached/new preprocessing
tokens separately. Valid reward=0 remains terminal; infrastructure failures have
reward=null and a failed phase. On explicit recovery, completed Solver artifacts
are reused for scoring. An interrupted Solver without a completion receipt is
never silently sampled again. Existing experiment roots and outputs stay intact.

`finish_index_failures.py` is a one-shot continuation for the approved UTF-8
signature-prefix fix. It waits for the original controller to exit with a complete
summary, then selects only header-decoding failures with no model request or
Solver invocation. Recovery uses a separate fixed code/run directory, the same
model/configuration/official scorer and three slots. Valid scores and cases that
already reached a model are not resampled. The final composite retains original
results/provenance and does not alter official rewards. Its status receipt also
records an interrupted parent or a failed continuation instead of silently waiting.

## Artifacts and recovery

`manifest.json` pins original issue, repository, prompts, backend and configuration.
`repository-graph.json` includes every eligible file plus parse/exclusion records.
`repository-outline.txt`, `targets.json`, `analysis/NN/retrieval.json`, raw requests,
raw responses, parsed analyses, retained analyses and `enhanced-issue.md` make every
stage auditable. Raw responses retain reasoning and original usage fields.

Rerunning with identical inputs reuses completed responses. Changed inputs/config
require a new output directory. A request lacking a response is not automatically
sent again. Result status, failed stage, per-call usage, new requests and new token
totals distinguish cached reuse from new expenditure. Completed preprocessing is
not a benchmark reward and should not be counted as a solved task.

## Checks

```bash
python -m pytest tests/test_codescout.py -q
ruff check external_agents_eval/preprocessors scripts/deepswe_codescout tests/test_codescout.py
```
