# V6.5 跨模型审查：GPT-6.1 Sol 审查，DeepSeek 求解与修复

与 V6.5（`scripts/deepswe_v65/`）完全相同，只换审查模型：

- X、D、I、R、S 以及 JSON 格式修正：GPT-6.1 Sol，Responses API，`reasoning.effort=xhigh`，最大输出 65536，`store=false`；两个主机侧端点按题轮换初始端点、故障切换（沿用此前GPT实验的 `routed_model.py`）。
- Solver（复用已有P0）与 Repair：DeepSeek V4.1 Flash，参数与 V6.5 相同。
- 提示词、文件交付合同、审查视图、信息屏障、D 通道、45 分钟窗口、G、评分：与 V6.5 相同。
- 每个角色调用的 `mini-invocation.json` 记录实际服务的模型；GPT 路由记录在 `endpoint-routes.jsonl`。

```bash
PY=/root/ft_m1_review-code-20261029-r1/.venv/bin/python
"$PY" scripts/deepswe_v65_gpt/run.py prepare --root <新目录> \
  --ledger-source /root/ft_m1-deepswe-m0failed8-20261028-w3-r1 \
  --ledger-source /root/ft_m1-deepswe-m0passed22-20261028-w3-r1 \
  --solver-source /root/v64-deepswe-eval30-20261101-w3-r3 --task-id <task> ...
"$PY" scripts/deepswe_v65_gpt/run.py preflight-view --root <新目录>
# DeepSeek（Repair）密钥经环境变量注入；GPT 端点凭据只从主机侧私有文件读取，不打印
GPT_REVIEW_ENDPOINTS_FILE=/root/.config/external-agents-eval/ft-m1-gpt61sol-20261030/endpoints.json \
  "$PY" scripts/deepswe_v65_gpt/run.py run --root <新目录>
```

本变体只新增文件（`external_agents_eval/agents/mini_v65_gpt.py`、本目录、`tests/test_mini_v65_gpt.py`），不修改 V6.4/V6.5 代码；回退时删除这些文件即可。
