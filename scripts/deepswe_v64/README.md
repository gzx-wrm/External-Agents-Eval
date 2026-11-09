# V6.4：DeepSWE eval30 完整重跑

流程：冻结FT清单 → 原生Mini Solver → G(P0) → I/R → S → 路由 → Repair → G(P1) → 原版DeepSWE评分。

- 清单：沿用ft_m1两批的冻结FT清单（不重跑A，不改内容）；Solver只在任务文本后追加英文版清单说明（`ledger_input_v64.py`），官方`mini.yaml`、模型参数和预算与M0/ft_m1一致。
- 审查：`MiniV64ReviewAgent`（V6.4配置：审查提示v5-6、控制器v5-5），I增加四条测试规则；I/R使用单一文件交付合同并写明材料与时间限制；Repair、S使用各自的系统提示；S正文不变。
- 构建检查G：在任务镜像的全新离线容器中只编译/收集测试，命令按生态约定从基线仓库识别；差分判定，base失败、超时或无适配即“不可用”。P0失败时强制repair并把日志交给Repair；P0可构建而P1不可构建时交付P0。I/R/S看不到G的结果。
- 每题一个槽位跑完全程，并发3；P0在所有模型阶段结束后另行隔离评分，仅作分析，官方结果以最终交付补丁为准。

```bash
PY=<含mini-swe-agent 2.4.6的venv>/bin/python
"$PY" scripts/deepswe_v64/run.py prepare --root <新目录> \
  --ledger-source /root/ft_m1-deepswe-m0failed8-20261028-w3-r1 \
  --ledger-source /root/ft_m1-deepswe-m0passed22-20261028-w3-r1
"$PY" scripts/deepswe_v64/run.py preflight --root <新目录>        # 零API：原生协议与输入核对
"$PY" scripts/deepswe_v64/run.py preflight-gate --root <新目录>   # 零API：30题G(base)/G(ft_m1 P0)
# 启动前仅向进程环境注入 DEEPSEEK_API_KEY，不保存到文件
"$PY" scripts/deepswe_v64/run.py run --root <新目录>
```

`prepare`之后源码被哈希冻结；`run`前任何源码改动都会被拒绝。已有`result.json`的题不会重跑；已有Solver输出时不会重新采样。
