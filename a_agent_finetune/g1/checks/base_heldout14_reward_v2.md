# G1 放宽奖励后的 base 历史输出复评

复评 Qwen3.5-9B base 最近一轮14题。使用原始最终 assistant 回答和实际终止状态，不使用历史提取、修正或编号后的 Ledger。

严格规则：1/14（7.14%）；放宽规则：9/14（64.29%）。新增8题得分来自允许说明文字和 Markdown 代码块，未修改清单内容。

| 题目 | 严格奖励 | 放宽奖励 | 放宽规则判定 |
|---|---:|---:|---|
| abs-stepped-slices | 0 | 1 | delivered |
| aiomonitor-task-snapshots-diff | 0 | 1 | delivered |
| cliffy-config-file-parsing | 0 | 0 | length |
| expr-try-catch-errors | 0 | 0 | timeout |
| fastapi-implicit-head-options | 0 | 1 | delivered |
| gql-incremental-graphql-delivery | 0 | 1 | delivered |
| meriyah-explicit-resource-declarations | 0 | 0 | timeout |
| numba-stencil-boundary-modes | 0 | 1 | delivered |
| obsidian-linter-auto-table-of-contents | 1 | 1 | delivered |
| sqlite-utils-safe-import-checkpoints | 0 | 1 | delivered |
| tengo-callable-instance-isolation | 0 | 1 | delivered |
| testem-per-launcher-reports | 0 | 0 | length |
| ts-pattern-match-each | 0 | 0 | length |
| yaegi-go-embed-directives | 0 | 1 | delivered |

长度截断3题、超时2题仍为0分。此结果是历史输出复评；旧推理环境及预算与 G1 不同，不作为同条件 G1 baseline。奖励不验证需求覆盖率或引用真实性。

逐题原始轨迹指针、奖励源码校验值和终止状态见 [JSON报告](base_heldout14_reward_v2.json)。
