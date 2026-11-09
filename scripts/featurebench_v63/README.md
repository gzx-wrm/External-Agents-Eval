# FeatureBench × Mini V6.3 单题接入

此接入不修改现有 V6.3 Agent 源码、提示词、证据整理器、schema、阶段预算和控制器。
首轮范围为 CPU-only Level 1。Level 2 与 GPU 题不在本次已验证范围内。

服务器独立目录：`/root/v63-featurebench-20261028`。

- 数据：`LiberCoders/FeatureBench`，v1.1 / `fast`。
- 数据快照：`76b4a4566e04f4bcc13c35125d4f301791efa736`。
- 官方代码：`8d4e347ec57546685c5a87e8676bf575db022ea6`。
- 单题：`pypa__packaging.013f3b03.test_metadata.e00b5801.lv1`。
- 这是接入冒烟，不是随机性能样本。

## 环境及任务基线

任务容器沿用 2 CPU / 6 GiB、network=none、cap_drop=ALL，无宿主目录与 Docker socket 挂载。
模型继续由宿主机现有 mini-SWE-agent 2.4.6 环境调用；官方评分单独使用新建的 Python 3.12 环境。

初始化直接调用所固定官方版本的 `_initialize_level1(white_box=False)` 和
`clear_package_caches()`，保留其仓库恢复、mask、F2P 删除、环境修复及字节码清理。
不调用安装 tmux/asciinema 的终端录制步骤；本接入不需要容器内 Agent CLI。
随后对**官方已构造完毕的 HEAD**执行单提交清理，不重置到数据集中原始的 `base_commit`。

Agent 仅收到公开任务描述。包含 patch、test_patch、F2P/P2P 的完整记录留在宿主机
`manifests/task-private.json`，不挂载、不上传给 Agent。初始化使用的 mask patch 会在进入 Agent 前删除。

环境桥接每次执行时激活 `conda testbed`，并将当前工作树及其 `src` 放到 Python 导入路径前方。
冒烟首先验证每个角色工作树中 `packaging.__file__` 的实际来源。

用户已确认的唯一工具路径适配：在 FeatureBench 的 A 文件工具环境中，将题面
`/testbed/...` 映射为当前 A 工作树下的同一路径。仍调用原 Pi 工具，原始请求保留在
原生 mini 轨迹中，转换后的工具记录同时保留 `featurebench_requested_path` 与实际
`path`，后续原证据整理器不变。原生 Bash 不做字符串替换。
离线验证使用真实 Pi `read` 分别读取相对路径和绝对路径，并要求两次成功读取都进入证据来源记录。

## 评分与记录

直接调用固定官方版本的 `featurebench.harness.run_evaluation.run_instance`。
评分在新的独立容器中执行，white-box 关闭。Agent 与官方测试不共享运行时容器。
先验证官方 gold 补丁通过及只增加无关文本文件的 control 补丁失败；此步骤无模型请求。
正式冒烟保存 `P0.patch`、`P1.patch`，分别运行官方评分，保留每阶段完整报告。

`%Passed` 采用官方每题 F2P 通过比例；多题时对各题比例求均值。
`%Resolved` 使用官方 F2P/P2P 完整通过判断。进程失败与评分异常单独记录，不作为正常解题失败解释。

## 执行

宿主 Agent 环境沿用已有的 `venv312`，不安装或升级其依赖：

```bash
ROOT=/root/v63-featurebench-20261028
export PYTHONPATH="$ROOT/project:$ROOT/vendor/FeatureBench"
/root/v62-swe-verified100-20261026/venv312/bin/python \
  "$ROOT/project/scripts/featurebench_v63/smoke.py" preflight \
  --root "$ROOT" --run "$ROOT/runs/preflight-packaging-r1"

# 确认 preflight/readiness.json passed=true 后，向进程环境注入 DEEPSEEK_API_KEY。
/root/v62-swe-verified100-20261026/venv312/bin/python \
  "$ROOT/project/scripts/featurebench_v63/smoke.py" generate \
  --root "$ROOT" --run "$ROOT/runs/smoke-packaging-r1"
```

同名运行目录不自动覆盖已有结果。查看运行目录的 `status.json`、`launcher.log`、
`agent/v6_review_repair/` 和 `scoring/P0`、`scoring/P1`。

服务器已启动单次接续程序 `supervise.py`：等待指定官方镜像就绪后执行 preflight，
仅当金标和负对照通过时启动一次完整模型解题，随后分别评分 P0/P1。
此程序不重复采样，不是定时监控，也不会创建额外批次。
实际运行目录为 `runs/smoke-packaging-20261028-r1`，前置验证目录为
`runs/preflight-packaging-20261028-r1`。

本轮代码验证：服务器 37 项通过，含真实 Pi 工具的路径映射与 FT 证据合并检查。
服务器缺少历史训练快照资料的 4 项比对已在本机执行并全部通过；不是四项运行逻辑失败。
官方镜像下的完整链路是否通过，应以 `preflight/readiness.json` 与最终评分结果为准。
