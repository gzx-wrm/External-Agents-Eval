# V6.5 回退到 V6.4

V6.5 几乎全部是新增文件；只改了两个已有文件，且改动不影响V6.4行为（有测试固定）：

| 文件 | 改动 |
|---|---|
| `external_agents_eval/workflows/solve_review_repair/v5_contracts.py` | `validate_review_batch` 允许角色 `D` |
| `external_agents_eval/workflows/solve_review_repair/v5_controller.py` | 提示词构建改走 `_lane_prompt` / `_summary_prompt`，默认实现就是原来的 `reviewer_prompt` / `summary_prompt` |

改动前的原文件备份在 `.runtime_cache/v65-pre-change-originals/`（含 `MANIFEST.json` 哈希，与服务器2 `/root/v64-code-20261101-r3` 一致）。

## 只想继续跑V6.4

不需要回退代码：服务器2上 V6.4 一律使用冻结快照 `/root/v64-code-20261101-r3`，V6.5 使用自己的快照目录，两者互不影响。

## 本地完全回退

```bash
cp .runtime_cache/v65-pre-change-originals/external_agents_eval/workflows/solve_review_repair/v5_contracts.py external_agents_eval/workflows/solve_review_repair/
cp .runtime_cache/v65-pre-change-originals/external_agents_eval/workflows/solve_review_repair/v5_controller.py external_agents_eval/workflows/solve_review_repair/
```

然后删除下列 V6.5 新增文件（只属于V6.5，V6.4不引用）：

- `external_agents_eval/workflows/solve_review_repair/review_view.py`
- `external_agents_eval/workflows/solve_review_repair/v65_config.py`
- `external_agents_eval/workflows/solve_review_repair/v65_prompts.py`
- `external_agents_eval/workflows/solve_review_repair/v65_contracts.py`
- `external_agents_eval/workflows/solve_review_repair/v65_state.py`
- `external_agents_eval/workflows/solve_review_repair/v65_controller.py`
- `external_agents_eval/workflows/solve_review_repair/skills/v6.5-review-note-files/`
- `external_agents_eval/agents/mini_v65.py`
- `scripts/deepswe_v65/`
- `tests/test_review_view.py`、`tests/test_v65_controller.py`、`tests/test_v65_prompts.py`、`tests/test_mini_v65.py`

验证：两个已有文件的 sha256 应与 `MANIFEST.json` 相同，`tests/test_v5_controller.py tests/test_v64_*.py tests/test_mini_v64.py tests/test_deepswe_v64.py` 全部通过。
