# featurebench_v6.4_woG：FeatureBench Lite 27 题

流程：A（V6.3 Mini A，逐题一次采样后冻结）→ 原生 Mini Solver（题面追加冻结清单）→ I/R → S → 路由 → Repair → 官方 FeatureBench 评分。本次不带构建检查 G（woG）。

- 题目：服务器 1 上原生 Mini（M0）已正式评分的 27 题（24 道 Lv1、3 道 Lv2）。排除 TRL（GPU 与上游模型下线）和两道 Liger（GPU）。
- 审查：V6.4 的审查提示词 v5-6、控制器 v5-5，`build_gate_enabled=False`。和 DeepSWE 的 V6.4 相比，只有基准适配上的差别：基线路径用 `/testbed`；每个角色的工作树都通过 conda testbed 环境和 PYTHONPATH 导入自己的代码。
- A：沿用不改的 V6.3 Mini A（Pi 文件工具 + FT 证据，`/testbed` 路径映射）。ledger 入口在 A 结束、Solver 开始前停下。A 失败时按 V6.1 规则在 P0 前终止，计为框架失败。
- Solver：原生 Mini，配置和 M0 逐项一致（启动时校验），只在题面后追加和 DeepSWE V6.4 相同的英文清单说明。
- 容器准备：A、Solver、审查都在原始镜像的新容器里，执行官方初始化和 Git 清理。每次都核对准备出的代码树与 M0 Solver 当时的一致。
- meson-python 可编辑安装（本批只有 pandas）：其导入钩子总把包指向 `/testbed`。A 和审查容器在准备工作树前先按钩子自带的构建命令更新构建，再按 meson 安装清单把构建产物（`.so`、生成的 `.py`）复制进每个角色工作树。不被 `.gitignore` 覆盖的产物写进仓库本地的 `info/exclude`，所以不会进入任何 diff。在角色工作树里执行的命令会设置钩子自带的跳过变量 `MESONPY_EDITABLE_SKIP`，在 `/testbed` 里执行的命令不受影响。局限：工作树里改动的 `.pyx` 不会重新编译。Solver 仍和 M0 一样直接在 `/testbed` 工作。
- 只承认 Solver 和 Repair 的修改：每个角色有独立工作树；审查角色的写入会被检测并还原；交付前 `/testbed` 会先还原，再只应用 P0 或 Repair 的结果。
- 评分：每题使用其 M0 当时的评分环境。批次 1 用官方默认设置（断网）；之后的批次用 CPU 设置；TestsFetcher 用 HF 镜像；Astropy erfa 注入 IERS 数据（A、Solver、审查和评分容器都注入）。参考和负对照直接复用 M0 的结果。最终补丁评分一次；P0 与最终补丁不同时另外评分一次，只作分析。
- 镜像：每组题跑完后，删除 Docker 中展开的镜像副本；`/mnt/featurebench-cache` 的 OCI 缓存永不改动。
- 并发 3；容器 2CPU/6GiB；按镜像分组推进。

```bash
CODE=<代码快照>; R=<新运行目录>
export DOCKER_HOST=unix:///run/featurebench-lite-docker.sock
export PYTHONPATH=$CODE:/root/v63-featurebench-20261028/vendor/FeatureBench
PY=/root/v62-swe-verified100-20261026/venv312/bin/python
$PY $CODE/scripts/featurebench_v64/run.py prepare --root $R          # 零 API
$PY $CODE/scripts/featurebench_v64/run.py preflight --root $R        # 零 API
# 付费入口只经 stdin 传入 key，不落盘：
$PY $CODE/scripts/featurebench_v64/run.py launch --root $R --entry ledger
$PY $CODE/scripts/featurebench_v64/run.py preflight-solver --root $R # 零 API
$PY $CODE/scripts/featurebench_v64/run.py launch --root $R --entry run --from solver
# 完整流程（含 A）：--entry run --from A
```

`prepare` 之后源码按哈希冻结。已有结果不会重跑，已冻结的清单和 Solver 输出不会重新采样。中断的尝试会改名保留为 `*.incomplete-*`。
