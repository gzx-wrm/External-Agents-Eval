# E 版提示词与 Pi → AutoDL Qwen 最小链路

`../e_prompt_v1.txt` 以 F 第四版为基础，保留 V6.1 的完整 O/P/Q 规则和
`SpecOutput` schema。修改限于运行场景：输入提供原始任务、
`managed_delivery` 和冻结仓库的 `base_root`，模型通过
`read / grep / find / ls` 自主获取证据。没有教师选择的证据包或清单。
工具探索期间可以调用工具；最终回答只允许一个原始 JSON 对象。

本目录包含独立的 Docker/Pi 接入测试，以及 `../run_e_inference.py`
正式 A 专用入口。两者都不启动 Solver、I/R、S、Repair 或 verifier。
仓库以只读挂载提供给 Pi，模型在 AutoDL 推理；工具执行、会话和轨迹
保存在本机。

## 已冻结的接入配置

- Pi：0.84.2；本地冒烟镜像安装 ripgrep 和 fd-find，提供可用的 grep
  与 find；Ubuntu 的 fdfind 通过 fd 别名接入 Pi。
- 模型：Qwen3.5-9B 原始 BF16 权重，revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，未加载 LoRA。
- 远端：vLLM 0.17.1，服务别名 `qwen35-9b`，Chat Completions API。
- 上下文：172,032；thinking 关闭；单并发；CPU prefetch offload。
- 工具解析器：实际安装版本支持 `qwen3_coder`；自动工具选择开启。
- Pi 自动上下文压缩关闭；模型声明的最大输出为 8,192。
- 前两次契约诊断使用 Pi/vLLM 默认采样。最终脚本通过一个显式 Pi 扩展
  固定 `temperature=0`、`seed=42`，与 F 原生基线一致。扩展仅修改采样
  参数并把实际 HTTP 请求正文保存到 `sessions/provider-requests.jsonl`，
  不修改模型回答、工具结果或要求清单，不记录鉴权头。
- 实际服务日志显示 prefix caching 关闭；多轮输入 token 是各轮输入之和，
  不能当成单次上下文长度。Pi 模型计价字段为零不代表 GPU 租赁免费。

## 启动和复现

新 AutoDL 主机连接入口为：

```bash
ssh -p 47076 root@connect.bjb1.seetacloud.com
```

`start_model.sh` 复用备份中的 `/root/autodl-tmp/a-agent-smoke/venv` 和
`model_path.txt`，另存为远端的 `start_model_e_v1.sh` 后启动。
模型服务仅监听远端 `127.0.0.1:8000`。密码不写入脚本或物料。

本机建立转发：

```bash
ssh -N -L 127.0.0.1:3000:127.0.0.1:8000 \
  -p 47076 root@connect.bjb1.seetacloud.com
```

Pi 配置使用 `http://host.docker.internal:3000/v1`，本机和容器都需要
能访问 `/v1/models`。E 的独立 Docker 入口不使用 Pier 出网代理；后续
DeepSeek 下游继续使用原有 Pier 环境和网络配置。

在项目根目录构建镜像并运行工具探针：

```bash
docker build -t external-agents-eval/pi-e-smoke:v0.84.2-arm64 \
  a_agent_finetune/e_smoke
bash a_agent_finetune/e_smoke/run_pi_probe.sh \
  <冻结仓库目录> <新的输出目录>
```

该 Dockerfile 面向当前本机的 arm64。运行脚本默认有 600 秒超时，
可用 `E_SMOKE_TIMEOUT_SEC` 调整。输出目录已存在 `pi.jsonl` 时拒绝覆盖。
仓库只读；Pi 的配置锁、凭据占位文件与会话使用独立可写目录。

`tool_probe.txt` 验证模型搜索、读取真实代码后继续回答，使用简单 JSON
诊断接口，不代表 E0 的 Ledger 成绩。`contract_task.json` 是另一个明确
标记的合成小任务，用完整 E 提示词测试 `SpecOutput` 最终输出；同样不
计入 30 题。运行脚本第三个参数指定 user 消息文件，第四个参数可指定
system 提示词文件。最终 E 接法沿用 F 的消息角色：完整 `e_prompt_v1.txt`
放在 system，原始任务和仓库入口放在 user。Pi 仅额外附加工作目录，
工具定义通过 API 的 tools 字段提供，不自动加载项目上下文文件。

本次输出位于
`../runs/qwen35-9b-e-minimal-20261003-host47076/`，其中也准备了完整原始
Aiomonitor 题目的 `spec-input.json` 和 `e-spec-prompt.txt`；准备输入不
代表已经完成该题的正式 A 评测。

每个运行目录保存 `prompt.txt`、可选的 `system-prompt.txt`、`models.json`、`settings.json`、
`pi.jsonl`、`pi.stderr.txt`、`sessions/` 和 `exit-code.txt`。
首次工具探针失败于 Pi 配置目录只读，未调用模型；第二次已改为独立的
可写配置目录，成功执行 grep、两次 read 和 ls。其原始最终回答带
Markdown 包装，审计保留该格式问题，没有把去包装后的内容冒充原始
JSON 合格结果。

契约诊断的前两次回答同样带 Markdown 包装，但都能通过 V6.1 现有的
单个 JSON 提取器和 SpecOutput 校验。该处理仅去掉外部包装，未调用
DeepSeek 或其他模型，也未修订内容。结果同时保留原始格式是否合格与
控制器提取后是否合格，不把二者混为一项。第一组契约测试的启动脚本
在运行期间被修改，结束记录出现 shell 错误；该组保留轨迹及错误说明，
不推断 Pi 的进程退出码。

`run_pi_probe.sh` 保留为接入诊断；正式 A 采样使用下面的新入口。批量
启动前，应先通过真实任务的完整 A 冒烟。生成的清单后续再从 Solver
接入 V6.1 下游。

### 正式 E0 入口复用与自纠错配置

新入口调用现有 `V6Controller.prepare_ledger()`，复用 V6.1 的
A 运行、SpecOutput 解析校验、JSON corrector、Ledger 编号、来源绑定
和 pre-solver 产物；仅在 A 完成后结束。工具运行通过 E 专用 Docker
adapter 完成，批次冻结仓库直接导出 Git blobs（保留 export-ignore 文件），通过
文件哈希和行范围绑定引用；不需要为 A 创建 Solver 等阶段的工作树。
没有复制 A 的解析／纠错控制逻辑，也没有修改共享 V6.1 核心代码。

`stage_model_config.json` 是现有 PiStageModelConfig 可直接读取的模型
配置，明确把 A 和 json_corrector 都绑定到相同的远端原生 Qwen 模型。
`controller_config.json` 的字段可直接用于构造 `V6Config`：格式修正
上限为一次，整轮 A 自动重试为零。新入口读取并实际应用两份配置；
旧的 `run_pi_probe.sh` 仍不读取它们、不调用 corrector。此配置只用于
A 采样阶段，下游 DeepSeek 实验使用其原有独立配置。

流程为首次 A 输出 → 现有确定性 JSON 提取和 schema 校验；仍无效时
最多调用一次无工具的 Qwen JSON corrector，之后再校验；仍无效则
记录失败。保留首次输出、修正输入输出、次数、token 和最终状态。
原有 JSON corrector 只允许修复格式和无歧义结构问题，不改义务含义、
证据或来源指向。因此 Markdown 包装若已可由原解析器提取，不触发
模型纠错；绝对引用路径、证据支持和义务遗漏也不应交给格式修正器
处理。E0 与后续 EM 使用相同的纠错规则，分别报告首次和纠错后结果。

运行新入口（在项目根目录，使用已安装项目依赖的 Python）：

```bash
python -m a_agent_finetune.run_e_inference \
  --repository <冻结公开仓库目录> \
  --task-input <仅含原始task和managed_delivery的输入JSON> \
  --task-id aiomonitor-task-snapshots-diff \
  --base-commit b73fea2e0682803bda7531c93cd1dfb360839175 \
  --output <新的结果目录>
```

2026-10-04 按用户要求更新：默认整轮 A（含可能的纠错）最多 3,600 秒，
不限制工具调用次数；`--max-tool-calls 0` 表示不限次数，仍记录全部调用。
不继续完整 A 重跑，不自动压缩或截断证据。上下文 172,032、单次输出
最多 32,768（包含思考），thinking=on。正式入口读取 `inference_config.json`，
A 与 corrector 共用冻结的采样参数：temperature=0.6、top_p=0.95、top_k=20、
min_p=0、presence_penalty=0、repetition_penalty=1、seed=42，并显式开启模板
thinking。服务端启用 qwen3 reasoning parser，Pi 模型元数据 reasoning=true。
Pi 原生设置 `httpIdleTimeoutMs=0` 关闭默认五分钟 HTTP 无数据超时，由
整轮 A 的硬超时统一控制；避免工具解析器缓冲长输出时把模型运行误判
为连接中断。该设置由 Pi 在初始化 dispatcher 时读取，不使用 Node preload。
采样参数使用 Qwen 官方编码场景思考模式推荐值，输出长度使用官方常规推荐值。
E0 与 EM 必须共用同一配置，与旧 F／独立探针及非 thinking E0 分开记录。
A 只加载显式 E 提示词和工具；corrector 无工具。输入采用
白名单，不转发标签或教师证据字段。工具路径限定为 `/work/base`。

批次入口 `run_e_batch prepare` 默认准备全部 30 题，不自动复用旧冒烟。
只有显式指定 `--reuse-smoke` 才复用已完成结果，并要求提示词、采样和预算
一致。旧非 thinking 批次保留其冻结 runtime 和结果，不混入本次 thinking 批次。

结果保留 `inputs/`、每次调用的原始 `pi.jsonl`、`output.txt`、实际请求、
会话、ATIF 轨迹和统计，以及 `manifest.json`、`v5/ledger.json`、
`v5/source-bindings.json`、`v6/pre-solver-ledger.json`、`v6/stages/A.json`。
后续可通过原有 fork plan 的 `start_at: Solver` 和 `reuse.A` 读取此结果。
输出目录必须是新目录，避免隐式复用已有 Ledger。引用为绝对路径时
保留原值并记 degraded，不静默归一化；有效 schema 不代表语义完整。

真实 Aiomonitor 冒烟的 r1 使用温度 0，在连续重复读取 `types.py` 超过
20 次后人工停止，所有输出与失败 manifest 保留在
`../runs/qwen35-9b-e0-aiomonitor-20261003-r1/`，不计作成功样本。r2 保持
提示词、任务与冻结仓库一致，仅切换到非 thinking 通用采样参数
（temperature=0.7、top_p=0.8、presence_penalty=1.5）进行诊断。这个比较
不能单独证明重复由采样造成。参数参考：
https://huggingface.co/Qwen/Qwen3.5-9B#best-practices。

r2 已完成：16 次模型调用、15 次工具调用、工具错误 0，551.5 秒生成
16 O／4 P／0 Q。原始回答有 Markdown 包装，经既有 fenced_json 提取
后通过结构校验；未调用模型 corrector。26 处引用有 4 处任务引用不是
连续原文，因此 Ledger 标记 degraded；没有修订原始输出。产物已通过
原 V6StageSource 的 Ledger 导入校验。详见
`../runs/qwen35-9b-e0-aiomonitor-20261003-r2/smoke-report.md`。

## 本次最终最小链路结果

`contract-t0-r3/result.json` 记录完整 E system 提示词、原始 user 任务和
固定采样后的合成契约测试：Pi 退出码 0，15 次模型调用、17 次工具执行，
read/grep/find/ls 均实际调用，工具错误为 0，最终 stopReason 为 stop。
仓库内容前后哈希一致；完整会话、实际请求和 ATIF 轨迹均已落盘。

清单有 7 条 O、4 条 P、3 条 Q，经 V6.1 现有 fenced_json 提取后通过
SpecOutput 校验，未调用模型修正器。原始回答仍带 Markdown 包装，且
9 条仓库来源使用了绝对路径；相对路径契约未满足，没有自动修改这些
来源。引用的任务文字、实际文件和行号范围可定位，不代表这些引用在
语义上支持相应条目。正式 A 入口应分别记录这些问题，再决定是否采用
可审计的格式或路径归一化。

本组累计输入 200,206 tokens，输出 2,622，单轮最大输入 18,783，缓存
读取为 0。累计输入包含多轮重复上下文，不是一次 200k 的请求。
这些数字只用于链路诊断，不能作为真实 Aiomonitor 题目的 E0 成绩。
