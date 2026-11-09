# A-Agent 微调实验物料

## 当前固定30题入口

从2026-10-20起，后续固定30题实验使用[`eval30_current.json`](eval30_current.json)：永久排除`sqlite-utils-safe-import-checkpoints`，改为`true-myth-iterable-collection-combinators`。可直接供F版推理使用的完整物料为`data/f_teacher_observed_30_current`（`--data-dir a_agent_finetune/data/f_teacher_observed_30_current --variant full`）。29道保留题的输入与原F版一致；替代题按同一教师实际可见证据规则构建。

替代题属于已有训练材料，题集继续用于开发实验。历史采样、训练数据和旧成绩保持原样；修订后的30题统计必须剔除SQLite、补入True Myth，不能直接沿用旧30题成绩。当前教师补充实验记录在[`runs/ft-ds41-replacement-true-myth-20261020-r1/README.md`](runs/ft-ds41-replacement-true-myth-20261020-r1/README.md)。

本目录先制作 F（固定证据）样本。选题及理由见 `selection.json`；五题均出自 V6.1 + DeepSeek V4.1 Flash 原始 30 题批次中直接通过的 17 题，因此也在后续的 27 题通过集合内。它们已被用于流程开发，只能作为开发样本，不能充当独立保留测试集。

运行 `python a_agent_finetune/build_f_pilot.py` 会读取既有 A 提示词、轨迹和 Ledger，从题目 `task.toml` 指定的公开仓库基线 commit 重新提取 A 实际打开过的**完整文件**。基线 Git 对象缓存在 `.cache/baselines/`，不纳入版本控制。每题原始记录写到 `data/f_pilot_5/records/`，供审核；面向 SFT 的 `prompt` / `completion` 写到 `data/f_pilot_5/training_candidates.jsonl`，每题一行。`manifest.json` 汇总来源、长度和自动校验结果。

此批标签是原 A Ledger 去编号后的**教师候选**。构建脚本已检查输出结构、题目引文和仓库引用的行号及证据可见性；对原 Ledger 中结束行恰好超出文件末尾 1 行的来源，候选版收窄到文件末尾，并在原始记录的 `source_corrections` 中保留原值及理由。尚需人工逐项审核这些修正及所有义务的语义支持、遗漏、原子性和 P/Q 裁决。审核前不要将它们视为金标准或正式论文成绩。完整文件证据规则避免了按目标答案倒推选择片段，但长度可能超过经济可行的训练窗口；待用实际学生 tokenizer 测长后，再制定统一的、与标签无关的缩减规则。

`measure_f_tokens.py` 接受所选模型的 `tokenizer.json`、模型名和权重 revision，写出 `data/f_pilot_5/lengths.json`。这里统计的是消息正文 token；正式训练前还需用**实际对话模板**复算完整序列和 loss mask。

`build_f_compact.py` 基于完整原始记录另生成 `training_candidates_32k.jsonl`、`training_candidates_64k.jsonl` 和对应片段记录。它优先保留 A 的聚焦读取，并用目标 Ledger 的仓库引用作为必留锚点；因此两套缩减证据是**训练流程试验条件**，不能直接作为保留测试集的证据生成办法。32K/64K 仅为正文 token 上限，脚本分别留出空间给对话模板；具体模型模板下仍须复核。

已用 Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` 的 `tokenizer.json` 与 `tokenizer_config.json` 在 `enable_thinking=False` 设置下复核完整对话长度；结果见 `data/f_pilot_5/chat_lengths.json`。本机下载的模型文件和临时 Python 包只存放在忽略的 `.cache/` 中。复现时先安装 `tokenizers` 和 `jinja2`，再运行两个测长脚本并提供对应的模型文件路径。

## 固定证据推理冒烟测试

`run_f_inference.py` 独立调用 OpenAI-compatible Chat Completions 接口，不依赖 Pi、Docker 或主评测框架。默认读取 32K 版的五题，按 `manifest.json` 顺序串行执行，每题仅发送 JSONL 中的 `prompt`，不发送教师 `completion`。它还会核对输入是否与对应的题目和证据记录一致，避免引用校验使用了另一套证据。

在 AutoDL 上只需上传本目录；不上传 `.cache/` 和已有的 `runs/`。保留脚本、`spec_output_schema.json` 和 `data/f_pilot_5/` 即可运行，不需要重新抽取样本或下载基线仓库。下方命令假设当前目录是上传后的 `a_agent_finetune/`。

### 启动模型服务

建议先用单张 48GB GPU 跑通。选择支持 vLLM 的 Python/CUDA 环境，在独立 Python 环境中安装：

```bash
python -m pip install "vllm>=0.17.0" "jsonschema>=4.23,<5"
export HF_HOME=/root/autodl-tmp/huggingface
```

只运行推理客户端的环境仅需安装 `jsonschema`。服务参数参考 [vLLM 官方 Qwen3.5-9B 指南](https://recipes.vllm.ai/Qwen/Qwen3.5-9B) 和 [Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)：

```bash
vllm serve Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --dtype bfloat16 \
  --language-model-only \
  --max-model-len 40960 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 127.0.0.1 --port 8000
```

客户端也会为每个请求指定 `chat_template_kwargs.enable_thinking=false`，与现有测长设置一致。服务启动并显示可接收请求后，在另一个终端、同一 Python 环境中运行客户端。两者都在 AutoDL 实例内运行，无需对外开放服务端口。

32K 版五题的实际输入是 27,391–28,417 tokens；默认输出上限为 8,192，所以上面为总上下文配置 40,960。如果选择 24GB GPU，先尝试将服务的 `--max-model-len` 改为 `32768`，并在客户端加 `--max-tokens 4096`；五题输入加这个输出预算均在该窗口内，能否载入仍以实际显存检查为准。脚本不自动裁剪输入，也不自动调整输出预算。

### 先跑一题，再跑五题

先运行第一题 `abs-stepped-slices`：

```bash
python run_f_inference.py \
  --limit 1 \
  --output-dir runs/qwen35-9b-smoke1
```

检查这一题的 `output.txt`、`validation.json` 和 `result.json` 后，再运行五题：

```bash
python run_f_inference.py \
  --output-dir runs/qwen35-9b-smoke5
```

五题命令会包含第一题，不复用之前的调用。每次实验使用新的输出目录，避免覆盖结果。默认 `temperature=0.0`、`seed=42`、请求超时 1,800 秒，适用于当前流程测试；正式能力比较时再统一采样条件。可用 `--variant 64k|full` 切换输入，但需要另外增大服务上下文和核算显存。使用其他服务地址或模型名时传 `--base-url`、`--model`；需要鉴权时在环境变量 `F_INFERENCE_API_KEY` 中设置，脚本不会把请求鉴权头写入结果。

### 输出和判断标准

每题写入 `runs/<实验名>/<task_id>/`：

| 文件 | 内容 |
| --- | --- |
| `request.json` | 实际发送的消息和生成参数 |
| `raw_response.txt`、`response.json` | 原始 HTTP 响应和解析后的接口响应；HTTP 错误只保存原始响应 |
| `output.txt` | 模型原始正文 |
| `spec_output.json` | 能按严格 JSON 解析时的输出对象 |
| `validation.json` | JSON、SpecOutput 结构、重复项、题目引文和可见仓库行号校验 |
| `result.json` | 状态、finish reason、接口 token 用量、耗时和错误信息 |

根目录的 `run.json` 记录运行参数、任务列表和数据/schema/脚本哈希；`summary.json` 在每题结束后更新，汇总状态、token 用量和耗时。token 用量沿用接口返回值，没有返回用量的失败请求不计入 token 总和。建议同时保存服务启动命令和 `python -m pip freeze` 的输出，以记录实际软件版本。

`passed` 只表示响应完整、JSON 与协议有效、来源锚点出现在输入中；**不表示义务在语义上正确、完整或达到教师水平**。脚本不与教师答案做自动相似度评分，不修补 JSON、不自动重试；有任一题请求失败、截断、未正常结束或校验失败时，进程退出码为 1，仍会继续保存其余题目的结果。随后应人工审核义务质量，才适合推进微调效果比较。

协议文件导出自 `v5_prompts.py` 的 `SPEC_OUTPUT_SCHEMA`，并补上原合约的空白文本、重复义务和整数行号规则。仓库内的 `tests/test_f_inference.py` 用本地模拟接口检查整个保存流程，不调用真实模型；当前脚本尚需在租赁 GPU 上完成真实模型冒烟。

## 30 题 Teacher 实际可见证据版（第四版提示词）

新物料位于 [data/f_teacher_observed_30_v4](data/f_teacher_observed_30_v4/README.md)。仅保留 A 的 read 工具实际返回且与冻结基线吻合的行，按实际截断范围提取并去重，不扩展成整份文件；提示词仍为 `f_prompt_v4.txt`。11 题采用人工标注、19 题采用原 A 标签，引用审查与前后修改记录随物料保存。

现有客户端用 `--data-dir a_agent_finetune/data/f_teacher_observed_30_v4 --variant full` 读取（在项目根目录调用脚本时）。全部输入约 2 万–16.2 万 tokens，预留 8,192 输出时只有 9 题适配 64k。不要沿用上面的五题服务长度直接启动全批；逐题长度、标签解释备注及复现方式见新目录的 README。本批仅完成物料制作与本地检查，未启动新的推理或训练任务。
