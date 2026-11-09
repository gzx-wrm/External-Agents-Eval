# V6.3 × DeepSWE 留出29题（eval30 同源流程）

用户 2026-11-02 确认的路线1：每题重新运行 A，然后接 ft_m1 的 Solver 和 ft_m1_review 的 v6.3 审查，与第一批 eval30 清单的生产和使用过程一致。
题目是留出30题去掉 wazero，共29题。wazero 在 M0 中因 git submodule 被冻结的 sanitizer 拒绝而成为基础设施错误，用户决定跳过。

## 每题流程

1. **A：Pi 探索 + FT 清单。** 沿用 V6.2 的 A 路径，代码不改：
   - `V62Controller._build_ledger`：先用历史 spec 提示词跑 Pi 0.84.2 探索，工具为 read/grep/find/ls；
   - 再冻结实际读到的基线文本作为证据，发一次无工具的 FT 请求（DeepSeek V4.1 Flash，thinking，high，max_tokens 65536，seed 42）；
   - `prepare_ledger` 的参数、stage id 和角色超时（1800 秒）均按 `PiV62SolveReviewRepairPierAgent.run` 构造。
2. **Solver。** ft_m1 的 `augment_task` 把冻结清单追加到原任务文本，然后用 M0 的原生 Mini 配置求解：
   - 配置与 M0 逐字一致：每题10800秒，无步数和费用上限，禁网，2CPU/8GiB；
   - 不输入证据包。
3. **P0 官方评分。** 在全新容器中进行，与 ft_m1 相同，只作分析用。
4. **审查。** 直接调用 `scripts/deepswe_ft_m1_review/run.py` 的 `review_case`：
   - FrozenMiniReviewAgent、MiniV63、V6 控制器，I/R → S → Repair/accept/hold；
   - 之后在全新容器中做 P1 官方评分。

模型全程为 DeepSeek V4.1 Flash（`deepseek-flash`），并发3。已有的终态结果直接复用。
A、Solver 有未完成的付费尝试时不会自动重采样，而是记为该题的阶段错误。

## A 容器与历史条件的对应

- **网络。** 历史 Pier 运行的 A 容器只能通过带认证的 Squid 代理访问 `api.deepseek.com`。本实验的做法：
  - 每题建一个 internal Docker 网络，放一个白名单代理容器（`egress_proxy.py`，仅用标准库，在同一个题目镜像中运行，不构建新镜像）；
  - 代理只允许 `CONNECT api.deepseek.com:443`，用户名 `agent`，口令为每题随机；
  - 代理环境变量与 `pier.environments.agent_setup.proxy_environment` 相同；
  - 宿主机不做任何网络配置修改。
- **Pi 运行时。** Node 22.19.0 来自官方归档，已核对 SHA256；Pi 0.84.2 用 `npm ci --ignore-scripts` 按固定 lock 安装。
  - 安装位置 `/opt/pi-runtime`，与 Pier 安装路径相同；
  - 环境变量与 `PiV6SolveReviewRepairPierAgent.run` 相同，含 `PI_OFFLINE=1`。
  - 历史 A 没有 rg/fd，grep/find 工具返回不可用，所以本运行时同样不带 rg/fd；
  - 清单见 `/root/v63-heldout-pi-runtime-20261102/runtime.json`。
- **仓库隔离。** 与 M0 的 `prepare_repository` 步骤相同：
  - 同一 sanitizer、同一合成基线提交，保留镜像环境改动，删除其他 `.git`；
  - 唯一不同是网络审计：改为检查直连被阻断、只有白名单可达、隐藏测试和标准答案不存在。
- **凭据。** API 密钥只通过宿主进程环境和 A 容器的 exec 环境传递，与历史 Pier 一致。轨迹和日志中会脱敏，不写入任何文件。

## 使用

代码快照目录内执行，Python 为 `/root/ft_m1_review-code-20261029-r1/.venv/bin/python`。

```bash
python scripts/deepswe_v63_heldout/run.py prepare --root <新目录> \
  --m0-root /root/mini-swe-deepswe-heldout30-20261102 \
  --pi-runtime /root/v63-heldout-pi-runtime-20261102/pi-runtime-a.tar.gz
python scripts/deepswe_v63_heldout/run.py preflight --root <新目录>
# 以下两步才调用模型；密钥从用户创建的 root-only 文件读入进程环境
python scripts/deepswe_v63_heldout/run.py smoke-a --root <新目录> --task-id <题目>
python scripts/deepswe_v63_heldout/run.py run --root <新目录> --workers 3
```

**preflight 做什么（0次模型调用）：**
- 核对29个镜像；
- 在第一题上实际搭建 A 沙箱，并完成：仓库隔离、出口策略探测、Pi 运行时安装；
- 用无效密钥走一遍真实的 PiRoleRunner 命令路径，确认 Pi 能经代理到达 API 并被拒绝认证；
- 用占位清单跑 M0 的离线协议，确认 Solver 输入只注入一次清单；
- 跑 `review_case(preflight=True)`，覆盖审查容器隔离、P0 转换和冻结 fork；
- 官方 gold/control 检查直接引用 M0 留出批次的 readiness。

**smoke-a：** 只跑一题的 A（Pi 探索 + FT），结果放在 `smoke-a/`，不计入批次，批次也不会读取。

## 输出

| 路径 | 内容 |
|---|---|
| `a/<task>/` | 隔离审计 `isolation.json`、代理日志、`v6_review_repair/`（Pi 轨迹、证据包、FT 请求/响应、`v6/pre-solver-ledger.json`）、`a-result.json` |
| `inputs/ledgers/<task>.json` | 冻结清单，即 envelope 中完整的 `ledger` 对象 |
| `inputs/<task>/task.json` | Solver 输入 |
| `solver/<task>/` | 原生 Mini 轨迹、P0、P0 官方评分 |
| `outputs/<task>/` | 审查全部产物、P1 和官方评分；`result.json` 中 `p0_reward` 为 P0 得分 |
| `result.json` | 汇总 |

## 范围与口径

- **重叠：** 29题中有18题属于 train60。但这里的 A 是 DeepSeek 教师模型，不是用 train60 微调的学生模型，所以不存在训练集内推理的问题。重叠情况见 `manifests/heldout30.json`。
- **对比对象：** M0 留出批次（原生 Mini，20/29）。本实验的 P0 相当于 ft_m1 臂，P1 相当于 v6.3 臂，二者都来自同一次 A 和 Solver 采样。
- **新增文件：** `scripts/deepswe_v63_heldout/`、`tests/test_deepswe_v63_heldout.py`。不修改任何既有代码；回退时删除这些文件即可。

## 2026-11-03：证据映射修复与2题恢复运行（用户已批准）

主批次中，happy-dom 和 sqlfmt 的 A 在冻结证据时失败（`Tool return differs from frozen baseline`）。原因是 Pi 0.84.2 的两种读法，F 协议的比较函数没有覆盖：

- **offset 小于 1：** Pi 按 `startLine = offset ? max(0, offset - 1) : 0` 处理，从第1行开始读；
- **CRLF 行尾：** Pi 只按 `\n` 分行，返回内容保留 `\r\n`。F 协议却用 LF 拼接的基线行做比较。

**修复方式：**
- 只改在线收集器 `ft_evidence.py`：调用 `returned_span` 之前，由 `pi_read_view` 把 offset 小于 1 的读取按 1 处理，并在基线含 CRLF 时把返回内容归一为 LF；
- `returned_span`（与 F 快照逐字相同）不变；
- 证据文本仍取自原始基线字节，provenance 记录模型实际发送的参数。

**验证：**
- `tests/test_ft_evidence_pi_reads.py`；
- `replay_evidence.py`：用已记录的 A 轨迹离线重放，确认主批次已有证据包逐字不变。

**恢复运行：**
- 用 `prepare --task-id ... --note ...` 在新目录里单独运行这2题，从 A 开始；
- 单独报告，不并入主批次的 pass@1；
- 改动前的原文件见本地 `.runtime_cache/ft-evidence-pre-pi-read-fix-20261103/`。
