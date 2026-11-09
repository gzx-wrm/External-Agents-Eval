# V6.5：审查视图（A）+ 假设抽取（C）+ 消歧通道（D）

流程：冻结FT清单 → 原生Mini Solver → G(P0) → 审查视图 → I/R ∥ X→D → S → 路由 → Repair → G(P1) → 原版DeepSWE评分。

与V6.4的区别只在审查阶段，Solver、模型、预算、G、评分不变：

- **A 审查视图**：I/R/S/D看到的是P0的“行为视图”：P0新增的注释和docstring被去掉（编译器/工具指令保留），P0对测试文件的改动被撤回。按`strip-all-tests → strip-new-tests → strip-comments`逐级尝试，P0能构建而视图不能时降一级，都不行就用原始P0并记录。I/R提示词与V5-6逐字相同，只插入一段REVIEW VIEW说明。Repair仍在真实P0上修改。
- **信息屏障**：审查开始前，审查工作树换成视图；容器里所有原始P0副本（`apply-<sha>.patch`、`/tmp/v65-*.patch`）被删除，不可达Git对象被prune。审查结束后扫描审查轨迹，结果写入`v65/barrier-audit.json`（仅诊断）。
- **C 假设抽取（X）**：唯一能读到被隐藏文本的角色。无工具、会话不上传到容器；把P0的解读性选择写成中性的行为陈述（不含理由，不得照抄注释），最多12条，失败不阻塞审查。
- **D 消歧通道**：目标为清单开放问题(Q)和X的陈述，最多6个，优先级：冲突型陈述 → 阻塞问题 → 解读型陈述 → 其他问题 → 未说明型陈述。对每个目标列出读法、构造判别探针、按证据等级（任务原文 > base仓库约定 > 通用知识）裁决；证据支持另一读法时注册一条派生义务（来源只能是任务原文或base仓库行），只有派生义务能被S送去Repair。
- **时间**：X与I/R并行，D在X结束后开始，三者共用I/R的45分钟审查窗口；构建视图在窗口开启前完成。

```bash
PY=/root/ft_m1_review-code-20261029-r1/.venv/bin/python
# 试点：只审查r3中5道失败题已有的P0（Solver输入逐字相同才复用）
"$PY" scripts/deepswe_v65/run.py prepare --root <新目录> \
  --ledger-source /root/ft_m1-deepswe-m0failed8-20261028-w3-r1 \
  --ledger-source /root/ft_m1-deepswe-m0passed22-20261028-w3-r1 \
  --solver-source /root/v64-deepswe-eval30-20261101-w3-r3 \
  --task-id meriyah-explicit-resource-declarations --task-id ...
"$PY" scripts/deepswe_v65/run.py preflight-view --root <新目录>   # 零API：真实容器内构建视图并跑G
# 启动前仅向进程环境注入 DEEPSEEK_API_KEY，不保存到文件
"$PY" scripts/deepswe_v65/run.py run --root <新目录>
"$PY" scripts/deepswe_v65/run.py summarize --root <新目录> --compare /root/v64-deepswe-eval30-20261101-w3-r3
```

主要产物（每题`outputs/<task>/review/agent/v6_review_repair/`下）：`v65/review-view.json`、`outputs/v65-view.patch`、`v65/assumptions.json`、`v65/disambiguation-targets.json`、`v5/exports/review-D.json`、`v65/barrier.json`、`v65/barrier-audit.json`。

回退方法见同目录`ROLLBACK.md`。
