# G1：从 Qwen3.5-9B base 开始的 Ledger 交付 GRPO-LoRA 实验

G1 是独立实验。从原始 `Qwen/Qwen3.5-9B` 初始化新 LoRA，不加载已有实验
adapter，不使用教师轨迹、教师 Ledger 或人工 Ledger 作为训练答案。
初始目标仅为：在固定预算内直接交付符合原始协议的非空需求清单。

物料 v5 使用 128k 轨迹预算及 CPU 中间张量卸载。A800 上已通过同长度单轨迹的
前向、反向传播和诊断参数更新容量检查；该检查不代表 Ledger 效果或完整训练收益。
正式训练的运行状态与输出见 `runs/a800-20261009/` 下的独立启动记录。

## 任务集与输入

沿用原 30 题开发题池的任务级 13/3/14 划分，清单见 `task_selection.json`。
沿用题池只是为了便于比较，训练目标、在线轨迹、模型权重和运行记录均属于 G1。
该开发题池已有历史使用，不声称是新的盲测集或仓库完全隔离的测试集。

| 划分 | 数量 | 用途 |
|---|---:|---|
| train | 13 | 在线生成候选并更新 LoRA |
| validation | 3 | 对照 base，比较 checkpoint |
| test | 14 | 选定 checkpoint 后评测，训练时不采样 |

`data/{train,validation,test}.jsonl` 每题只有原始任务、仓库身份和 system/user 输入。
原始任务从 benchmark 的 `instruction.md` 和 `task.toml` 导出；仓库使用缓存中
原始 base commit 的 Git blobs，不使用工作区修改、benchmark `tests/` 或 `solution/`。
公开仓库自身原有的测试属于可见证据，包含在快照内。

`data/archives/` 包含 30 个冻结仓库压缩包，总计约 94.4 MiB、22,604 个文件。
`data/manifest.json` 记录文件校验值、任务说明校验值、仓库 revision 和快照身份。
两项公共 gitlink 未展开：Adaptix 的 `benchmarks/release_data`、Tomlkit 的
`tests/toml-test`。目录保留为空，具体情况在 manifest 中逐题记录。

模型输入不包含目标清单、先前探索轨迹、隐藏测试、Verifier 信息或 reference patch。
任务正文完整保留；A 只负责抽取需求，不实施题目中的代码或 Git 操作。

## 唯一奖励

```text
R = 1：正常结束；最终回答中包含完整 JSON 对象；通过原始 schema；
       obligations 至少一项；必需文本非空白。
R = 0：其余模型结果，包括空清单、格式错误、循环耗尽预算、超时或截断。
```

允许首尾空白，允许 preservation_obligations/open_questions 为空。
允许 JSON 前后的说明文字和 Markdown 围栏；从左到右提取完整对象，
取第一份通过全部校验的清单。不补括号、不修改字段、不纠正输出、不重试。
只从最终 assistant 回答提取，不从中间消息或工具结果提取。
来源字段只检查结构，不检查引用真实性或语义覆盖。
不奖励条目数量、长度、教师相似度、工具调用次数、速度或下游解题结果。
一条与二十条合法义务奖励相同。语义质量应另行抽查，不能从奖励上升推断需求完整性。

后端、模型加载、快照或 CUDA 错误会让运行失败并保留已写轨迹，不转成奖励 0；
模型调用不存在的文件、未知工具或错误参数属于可见工具反馈，模型可继续探索。
整个候选最终是否交付仍是唯一评分依据。

## 在线采样与更新

训练入口为 `train.py`，策略优化由固定的 TRL 1.15.0 GRPOTrainer 执行。
G1 提供原始任务到多轮工具轨迹的采样边界。每条候选有独立对话和工具日志，
只读访问同一冻结快照，提供 read/grep/find/ls 四个工具。

G1 训练使用当前进程中的 Hugging Face actor 顺序生成候选；每次采样自然使用
最新权重。训练不依赖独立的 Pi/vLLM 推理服务。相应的 base 与 checkpoint 对照
也使用同一 G1 入口，因此不能将以前不同运行协议下的完成率直接当作本轮基线。
该初版没有跨候选的 prefix cache；生成速度和容量在 GPU 冒烟中测量。

保存实际采样的 assistant token IDs；追加工具观察和 assistant header 时掩码为 0。
不重序列化或替换模型已经生成的工具调用，不对 system/user/tool token 优化。
原始最终模型 turn 是奖励输入；奖励函数确定性提取完整 JSON，响应 parser 仅用于工具路由。
旧策略概率和参考策略概率由 TRL 从相应模型计算；一次候选组只进行一次更新。

| 参数 | v5 物料配置 |
|---|---|
| Base revision | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| 精度 / LoRA | BF16，r=16，alpha=32，dropout=0 |
| 起点 | 原始 base + 新 adapter |
| 每题候选 | 4，顺序采样，各自随机生成，不反复重置同一随机种子 |
| 训练 microbatch / 累积 | 1 / 4，一组四个候选对应一次更新 |
| 学习率 | 5e-6，预热 2 步 |
| 更新次数 | 26；13 个训练任务约两次在线访问，并非教师数据的两轮 SFT |
| GRPO loss | 原始 grpo，group reward normalization |
| KL 系数 | 0.02；参考原始 base，是更新约束，不是额外质量奖励 |
| 验证 / 保存 | 每 13 步；保存 optimizer 和随机状态，不自动上传 |
| Thinking | 关闭 |
| 采样 | temperature=0.6，top_p=0.95，top_k=20 |
| 每候选时间上限 | 1,800 秒，覆盖全部工具回合 |
| 最大完整上下文 | 147,456 tokens，为初始提示词额外预留 16k |
| 最大轨迹后缀 | 131,072 tokens，包含模型与工具/header tokens |
| CPU 中间张量卸载 | 开启；策略 loss 前向使用 `save_on_cpu(pin_memory=True)` |
| 模型累计生成上限 / 单回合上限 | 16,384 / 8,192 tokens |
| 工具次数上限 | 无；受总时间和 token 预算约束 |
| 单次工具输出 | 12,000 字符；read 最多 400 行，其余工具输出同样有界 |

达到预算即终止候选并得 0，不截去旧上下文继续，也不注入额外“停止探索”消息。
`mask_truncated_completions=false` 保留预算失败候选的模型 token，以便其负相对优势
能参与更新；所有工具 token 仍不参与 loss。组内全 0/全 1 的候选不提供区分信号，
记录 mixed_group_fraction 和 ledger_delivery_rate，先在小样本中观察信号是否足够。

CPU 卸载只改变反向传播所需张量的存储位置，不改变奖励、模型精度、LoRA 参数或
完整注意力上下文；与现有梯度检查点共同使用，换取更多 CPU 内存和数据传输耗时。

## 使用步骤

以下命令从项目或物料包根目录执行。正式训练不在本次物料制作中自动启动。

### 1. CPU 验证

Python 3.12，安装 `requirements-check.txt`，不需要权重或 CUDA。

```bash
python -m pip install -r a_agent_finetune/g1/requirements-check.txt
python -m a_agent_finetune.g1.train --check-materials
python -m unittest a_agent_finetune.g1.tests.test_materials -v
```

已有固定 revision 的 tokenizer 时，可额外运行真实工具协议检查（不加载权重）：

```bash
python -m a_agent_finetune.g1.check_tokenizer --tokenizer-path /path/to/pinned/snapshot
```

检查 30 题初始 prompt token 与原始模板一致、Qwen XML 工具参数解析正确、
追加工具结果时保留采样 token 且准确屏蔽观察 token。使用合成回答，不代表模型解题成功。

### 2. 准备 GPU 环境

在单独的 Linux CUDA Python 3.12 环境安装 `requirements-train.txt`。
机器必须有 ripgrep；Qwen3.5 的 DeltaNet 快速路径需要匹配 CUDA/PyTorch 的
flash-linear-attention、causal-conv1d。A800 是否能承受最坏长度以实际冒烟为准。
如果需要修改预算或依赖，建立新的物料版本，重新冻结 manifest，并重跑同条件 base。

不提供 `--model-path` 时，入口按固定 revision 从 Hugging Face 下载原始权重。
已有权重时，传入目录名为固定 commit 的原始 HF snapshot 目录；不能传 LoRA 目录。
完整多模态 checkpoint 只加载 text backbone；检查缺失权重，禁止用随机初始化的
未加载参数进行训练。模型权重不包含在约 95 MB 的物料包内。

### 3. 无更新地采样一题，检查二值奖励是否有区分

```bash
python -m a_agent_finetune.g1.train --mode evaluate --split train \
  --task-id termenv-preserve-ansi-resets \
  --output a_agent_finetune/g1/runs/base-one-calibration
```

生成四个候选、评分并保存轨迹，不执行参数更新。如果候选都得 0，应先检查
是否能在预算内交付；不通过增加语义奖励或人为改写答案制造成功。

### 4. 单题两步训练冒烟

```bash
python -m a_agent_finetune.g1.train --smoke \
  --output a_agent_finetune/g1/runs/smoke-1
```

使用 Termenv 训练题，两次更新并逐步保存 checkpoint；随后在一项验证任务上评测。
结果记录实际 LoRA 参数变化。全零优势可能导致参数没有变化，这不等于已验证训练收益。
另起进程重载 final adapter，确认保存与加载链路：

```bash
python -m a_agent_finetune.g1.train --mode evaluate --split validation \
  --task-id vulture-persistent-analysis-cache \
  --adapter a_agent_finetune/g1/runs/smoke-1/final \
  --output a_agent_finetune/g1/runs/smoke-1-reloaded
```

### 5. 正式训练与评测

```bash
python -m a_agent_finetune.g1.train \
  --output a_agent_finetune/g1/runs/train-1
```

先运行未训练 base 的 3 题验证，再执行 26 步更新，保留 step13、step26 和 final。
本轮按用户要求直接训练，启动时增加 `--skip-base-validation`，跳过训练前评测和
额外冒烟；训练中每 13 步与结束后的验证仍按既有配置执行。相同协议的未训练 base
对照因此尚未完整采集，后续比较模型收益时需要单独补齐。
按验证集无需修复的清单交付率选 checkpoint；并列优先较早 checkpoint。若均不优于 base，
如实报告没有交付收益。选定模型后才运行 test，不按测试结果反复选模型。

```bash
python -m a_agent_finetune.g1.train --mode evaluate --split test \
  --adapter a_agent_finetune/g1/runs/train-1/checkpoint-13 \
  --output a_agent_finetune/g1/runs/selected-test
```

示例中的 checkpoint-13 只是路径示范，必须按实际验证结果选择。
测试集的 base 对照使用相同命令去掉 --adapter，另建输出目录。

所有 --output 必须是新目录，不覆盖已有结果。当前入口保存完整 checkpoint，
但不提供自动恢复或后台调度，避免初版引入额外控制逻辑。

## 输出与审查

每条候选保存 trajectory.json（准确 token IDs、掩码、逐回合原文、工具记录）、
final.txt 和 reward.json。候选组保存 batch.json；训练目录保存实际协议、物料
manifest、result.json、trainer_state.json 和 checkpoints。
每次训练日志保存 metrics.jsonl、loss.csv，并更新 loss_curve.png；
CSV 同时记录交付率、混合奖励组比例、KL 和学习率。GRPO loss 不要求单调下降，
交付率和有效奖励差异才是判断本轮目标是否改善的主要依据。

主指标是无需修复的非空 Ledger 交付率；另记录组内混合奖励比例、失败类型、
生成量、耗时和显存。保留清单用于人工语义抽查，第一轮不加入奖励。

`checks/report.json` 标明本次实际完成的本地检查及尚未完成的 GPU 检查。
`dist/g1_materials_v5.tar.gz` 包含 128k 轨迹预算、CPU 卸载、允许 JSON 提取的奖励、
loss 曲线记录和加载诊断序列化修正，解包后保留模块路径。
旧的 v1/v2/v3/v4 物料包保留；v1 使用整段回答必须为纯 JSON 的严格奖励。
包内不含 Python 环境、GPU 模型权重、历史实验数据或已启动训练结果。

协议来源：训练模板和工具响应解析 schema 冻结自
[TRL v1.15.0](https://github.com/huggingface/trl/tree/v1.15.0/trl)。
算法与接口说明见 [GRPO 文档](https://huggingface.co/docs/trl/main/grpo_trainer)。
