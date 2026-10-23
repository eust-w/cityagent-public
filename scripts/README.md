# 辅助脚本

这些工具需要显式执行，不会随安装或离线示例自动启动。首次使用从根目录 [README](../README.md) 和 [usage](../docs/usage.md) 开始。每个脚本的参数、依赖和环境要求以源码及 `--help` 为准，不要直接批量运行本目录。

| 用途 | 主要脚本 |
|---|---|
| 模型与运行环境检查 | `prepare_model_identity.py`、`probe_contest_runtime.py`、`probe_engine_capabilities.py` |
| 文档准备及覆盖检查 | `inventory_visual_evidence.py`、`prepare_document_image_representations.py`、`benchmark_document_preparation.py` |
| 单次及多任务评测 | `prepare_generation_manifest.py`、`prepare_multitask_manifest.py`、`evaluate_multitask.py`、`summarize_multitask.py` |
| 执行及后处理审计 | `audit_agent_run.py`、`audit_generated_postprocessing.py`、`evaluate_document_artifacts.py` |
| 恢复与并发测试 | `benchmark_rsi_recovery.py`、`benchmark_rsi_process_recovery.py`、`probe_rsi_parallel_sdk.py` |
| 推理策略实验 | `run_meta_optimization.py`、`run_rsi_evolution.py`、`propose_rsi_strategy.py`、`probe_rsi_sandbox.py` |
| 结果查询及输出 | `report_qa.py`、`render_states.py` |
| 显式数据与模型准备 | `prepare_training_revisions.py`、`build_sft_dataset.py`、`token_stats.py`、`train_lora_nvidia.py`、`train_lora.py`、`merge_lora_v3.py` |
| 部署包检查 | `package_submission.py`、`smoke_submission.py` |

数据准备、训练和合并工具作为可选工具保留，正常推理与 RSI 不会自动训练模型。模型、数据和第三方 SDK 均需自行准备。名称中含硬件或赛事约定的工具可能依赖特定目录、驱动或 SDK，不能直接作为通用部署命令。

完整入口与执行边界见[训练指南](../docs/training.md)、[量化指南](../docs/quantization.md)和[实验指南](../docs/experiments.md)。现代 NVIDIA trainer 默认会训练，先使用 `--prepare-only`；兼容单设备入口默认只准备，只有 `--execute` 才训练。

评测工具应使用独立运行目录和显式数据清单。生成接口不得读取评测目标标签。恢复工具可能终止它自己启动的测试进程，应先阅读脚本说明并在独立测试环境运行。打包工具只生成和检查本地产物，不代表满足任何外部平台的当前提交规则。
