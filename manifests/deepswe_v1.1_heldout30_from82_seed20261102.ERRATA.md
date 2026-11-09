# Errata (2026-11-02)

`overlap.note` in `deepswe_v1.1_heldout30_from82_seed20261102.json` says train60 tasks
"were used to fine-tune the FT A-agent". That is imprecise:

- train60 (`training_only`) is training material for the fine-tuned student A-agent
  (Qwen3.5-9B, `a_agent_finetune/`). Only experiments that use that student model are
  contaminated on these tasks.
- The frozen "FT" ledgers used by ft_m1 / v6.4 / v6.5 on eval30 were produced by DeepSeek
  V4.1 Flash (thinking, tool-free) from the fixed teacher-observed evidence packet ("F"
  version); no fine-tuned model was involved. DeepSeek-based A/F ledgers on train60 tasks
  are not training-contaminated.

The manifest itself is left unchanged because completed runs recorded its hash.
