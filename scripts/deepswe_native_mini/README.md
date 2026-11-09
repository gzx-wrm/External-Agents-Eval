# Native mini-SWE-agent × DeepSWE FT30

服务器2：`root@223.109.239.36:12748`。
独立目录：`/root/mini-swe-deepswe-ft30-20261028`。

本轮使用 `a_agent_finetune/eval30_current.json` 的30题，顺序不变，排除SQLite、包含True Myth。
30份任务目录校验和逐题匹配 FT 实际结果的 `task_checksum`，所有300个任务文件转移后再次检查SHA-256。
保留True Myth替代题与既有训练材料重叠的来源说明，不据此改选题。

## 用户确认配置

- mini-SWE-agent 2.4.6，原生 `DefaultAgent`、`DockerEnvironment` 和 Bash。
- 逐字保留官方通用 `config/mini.yaml` 的 system/instance/observation/format-error 模板。
- DeepSeek V4.1 Flash，服务别名 `deepseek-flash`，high，thinking enabled，最大输出65536。
- 每题10800秒，无步数停止和费用停止阈值，每题一次正式采样。
- Bash沿用原生Docker环境的默认30秒命令超时；模型API超时和重试保留原生默认。
- 官方任务资源：2 CPU / 8192 MiB；官方评分超时1800秒。
- 并发3，每个槽位完成“解题 → 独立官方评分 → 本题镜像清理”后再接下一题。
- 不提供A输出、固定证据、清单、审查建议，不增加Reviewer或Repair。

沿用此前已确认的唯一模型兼容修正：发送请求时将显式空字符串 `reasoning_content`
转换为一个空格；不修改原始会话中的值，不为缺失/None字段补造推理。

## 环境与补丁

解题容器禁网，不挂载宿主目录、评分目录或Docker socket，不转发API凭据。
模型请求由宿主机发出。

先记录镜像在原始base commit之上的环境兼容改动，再清理Git为单个合成基线提交，
重放环境改动并核对一致。仅为合成基线增加`main`别名，满足任务中的分支交付约定；
不保留原历史提交、远程或不可达历史对象，其他副本的Git历史也会删除。

官方通用Mini通过终止命令结束任务，交付物是工作目录中的修改。
评测层使用临时Git index导出相对合成基线的完整变化，包含已提交、未提交和未忽略的新文件，
不改变真实index，不增加修复或模型调用。补丁在原始任务base tree上可应用，交给官方grader。
达到原生运行限额时保留其退出状态和现有代码，再独立记录最终产物分数。

## 官方评分

在新的原始任务镜像容器中，注入原版 `test.sh/test.patch/grader.py/config.json` 和模型补丁，
执行原版 `/tests/test.sh`。这与30份官方tests/Dockerfile的四个COPY及chmod步骤对应，
不构建额外评分镜像，不改变隐藏测试、白名单、F2P/P2P映射或reward计算。
解题容器在评分开始前销毁，Agent无法看到评分测试或反馈。

预检使用第一题SQL Formatter：

- 原生Mini模拟响应离线检查通过，验证Bash、终止、补丁导出及推理字段兼容；0模型API调用。
- 官方gold：reward=1，F2P 26/26，P2P 5709/5709。
- 未改代码负对照：reward=0，F2P 0/26，P2P 5709/5709。

## 状态文件

- `launch.json`：启动PID、冻结配置、源文件hash；不含API key。
- `monitor-status.json`：30题逐题阶段。
- `outputs/<task_id>/native.traj.json`：原始原生Mini轨迹。
- `outputs/<task_id>/isolation.json`：环境与Git隔离审计。
- `outputs/<task_id>/model.patch`：交付补丁。
- `outputs/<task_id>/evaluation/verifier/`：官方reward、CTRF与原始测试日志。
- `outputs/<task_id>/result.json`：分数、退出状态、异常分类与补丁hash。
- `result.json`：本轮全部槽位终结后的汇总。

正常reward=0、缺少有效官方分数的评分故障、模型/流程异常保持分开。
不会把缺少reward的异常算作正常失败，不会自动重复已有模型采样。

首批Expr和Scriggo在ECR认证时EOF，均未进入模型。已启动单次接续程序
`finish_image_cases.py`：等待当前主轮全部退出后，仅处理`images/create`阶段失败、
且没有原生轨迹或预测的题目。先重新准备官方镜像，再保留并归档原故障记录，
使用同一30题清单和同一配置补上这些题目的首次模型采样。
已有模型采样或有效分数不会重跑，主轮与接续不重叠，并发上限仍为3。
接续状态见`continuation-status.json`，故障历史见`infrastructure-history/`。

## 三题 Git 隔离恢复

独立恢复目录：`reruns/isolationfix3-20261028-r1`。
仅重新启动此前尚未进入模型的 Adaptix、Tomlkit、Numba；保留全部原始结果。

- Adaptix/Tomlkit：保留子模块文件、文件模式、符号链接和父仓库gitlink；删除子模块Git元数据，
  不向父仓库对象包复制子模块commit。逐个比较隔离前后的公共文件快照，并保持父仓库tree一致。
- Numba：原失败仅由diff index哈希缩写长度变化造成，源码SHA-256实际一致。
  快照命令增加`--full-index`，继续逐字校验完整diff，不放宽基线一致性要求。
- 增加`--task-id`以限定恢复范围，完整30题清单和任务校验和不变。
- 除快照命令和CLI选题外，原生Agent调用、提示词、模型参数、预算及评分函数均保持不变。

验证包括两项Git对象/子模块快照回归检查，以及三题在原任务镜像内的原生Mini离线循环、
补丁导出、禁网/单基线审计和实际模块导入。三题均通过，验证期间模型API调用为0。

恢复任务仍为并发3、解题后立即评分。完成后在原实验目录生成`composite-current.json`，
用本轮三题结果与其余27题合并；不覆盖历史`outputs`和原始失败记录。

源脚本入口：

```bash
ROOT=/root/mini-swe-deepswe-ft30-20261028
PYTHON=/root/mini-swe-verified100-20261027/venv312/bin/python
"$PYTHON" "$ROOT/scripts/run.py" prepare --root "$ROOT" --workers 3
"$PYTHON" "$ROOT/scripts/run.py" preflight --root "$ROOT" --workers 3
# 启动前仅向进程环境注入DEEPSEEK_API_KEY，不保存到文件。
"$PYTHON" "$ROOT/scripts/run.py" run --root "$ROOT" --workers 3
```
