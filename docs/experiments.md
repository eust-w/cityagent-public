# 实验与评测

本页说明如何用合成样例或自己获准使用的数据开展实验，以及现有工具能证明什么。基础安装不包含模型、LoRA、真实报告、标签、比赛 SDK、CUDA/PPU 运行时或历史实验产物。下方涉及模型的命令是待配置的运行方法，不是本次公开版本已经取得的质量或性能结果。

训练准备见 [训练文档](training.md)，权重合并与 W8A8 导出见 [量化文档](quantization.md)。这里重点讨论固定模型后的抽取、评分和推理策略实验。

## 先运行不需要模型的检查

从仓库根目录安装开发环境后，生成纯虚构 DOCX 并运行离线流程：

```bash
uv sync --locked --extra dev
demo_dir="$(mktemp -d)/cityagent-example"
.venv/bin/python examples/offline_demo.py --output-dir "$demo_dir"

.venv/bin/python -m pytest -q \
  tests/test_generation_manifest.py \
  tests/test_summarize_generation_evaluation.py \
  tests/test_multitask_raw_output_contract.py \
  tests/test_multitask_end_to_end_latency.py \
  tests/test_training_revisions.py \
  tests/test_rsi_search_release.py
```

离线示例检查输入解析、规则抽取和结果文件；这些测试用合成记录检查分母、原始输出、来源修订和发布约束。它们不会调用模型，也不能证明真实长报告的语义效果、GPU 吞吐或故障恢复时间。

评测工具的帮助可以在基础环境中查看，例如：

```bash
.venv/bin/python scripts/evaluate_generation_nvidia.py --help
.venv/bin/python scripts/evaluate_multitask.py --help
.venv/bin/python scripts/run_rsi_evolution.py --help
.venv/bin/python experiments/prepare_ppu_quantization.py --help
```

`--help` 成功不表示已经安装执行阶段需要的依赖。实际 tokenizer 操作需要兼容的 Transformers；NVIDIA 生成需要对应推理依赖和显存；API 评测需要已经启动的兼容服务以及本地模型身份文件。PPU 环境应保留厂商 torch/vLLM，不能直接用开发或训练依赖覆盖。

## 选择评测协议

| 路径 | 输入和评测单位 | 主要用途 | 不包含的结论 |
|---|---|---|---|
| 单次生成 | 冻结 SFT 中每份整报告的一次模型生成 | 比较 adapter 的原始生成与保守融合结果 | 完整文档转换、并行流水线、发布和恢复验收 |
| 完整报告多任务 | 原始 DOC/DOCX，经整份或分块推理后每报告计一次指标 | 比较最终三表、失败、覆盖和真实调用记录 | 自动证明数据独立或官方成绩 |
| 参数元搜索 | 固定模型下筛选提示、预算、批量等配置 | 产生待复测候选 | 自动取得 RSI 发布资格 |
| 推理 RSI | 固定权重的策略包、隔离执行、真实复测和恢复测试 | 验证策略及有限搜索代码变化 | 训练、Judge 评分或无门槛在线自修改 |
| Kev / NER / PPU 探针 | 各自特定的材料、模型或硬件 | 研究单个抽取路径或设备能力 | 与主模型或其他设备直接可比的性能 |

同一个配置文件出现在两条路径中，不会使评测协议自动等价。必须固定源数据、样本分母、评分版本、模型和 tokenizer、提示、后端、硬件、预热与缓存条件，并为基线和每次候选运行使用独立目录。

## 单次生成：固定 2018 年留出协议

工具链是 `prepare_generation_manifest.py → evaluate_generation_nvidia.py → summarize_generation_evaluation.py`。输入 SFT 行需要非空的 `instruction/input/output`，以及明确的 `meta.year`。准备方法和目录格式见 [整报告 SFT](training.md#整报告-sft)。

当前协议固定选择全部 2018 年样本，训练长度过滤不能减少此处的评测分母。`--token-manifest` 必须来自 `benchmark_lora.py` 或现代 trainer 的完整准备清单，包含被过滤及其他年份行的审计信息。`--prepare-only` 仍需真实本地 tokenizer，并非无依赖操作。

以下示例使用自备资产和新的输出路径；生成命令会真正加载底座与 adapter：

```bash
python scripts/prepare_generation_manifest.py \
  --sft /path/to/authorized-sft.jsonl \
  --token-manifest /path/to/prepared-training/manifest.json \
  --output /path/to/new-generation-selection.json

python scripts/evaluate_generation_nvidia.py \
  --model /path/to/local-base-model \
  --adapter /path/to/matching-adapter \
  --sft /path/to/authorized-sft.jsonl \
  --selection-manifest /path/to/new-generation-selection.json \
  --partition diagnostic \
  --out-dir /path/to/new-generation-diagnostic

python scripts/summarize_generation_evaluation.py \
  --sft /path/to/authorized-sft.jsonl \
  --selection-manifest /path/to/new-generation-selection.json \
  --tokenizer /path/to/local-base-model \
  --generation-dir /path/to/new-generation-diagnostic \
  --partition diagnostic \
  --output /path/to/new-generation-summary.json
```

冻结清单按提示长度分桶选取诊断集，剩余样本组成 `final_acceptance`；小数据集可能没有后者。完成诊断后，在新目录中使用同一清单及 `--partition final_acceptance` 生成和汇总。`--generation-dir` 可以重复传入互不重叠分片的精确输出目录。汇总器保留完整分区分母，将缺失、失败和上下文超限作为没有预测处理，同时分别报告 `raw_model` 与 `model_preserving`。

这仍是内部年份留出：同一桥梁可能跨年份出现，2018 年也可能已经参与损失观察。反复根据 `final_acceptance` 调参后，它就成为开发数据。`--limit` 或 `--sample-sha256` 适合诊断，不能把子集完成说成完整验收。汇总只加载本地 tokenizer、无需模型权重，但仍需要相应依赖。

## 完整报告：绑定源文件和独立标签

`prepare_multitask_manifest.py` 将 SFT 标签绑定到原始 DOC/DOCX。通过文件名及可选 `meta.report_sha256` 找到唯一来源，生成 schema V2 清单及单独的 gold 文件。真实推理接口只接收报告；标签由评测器在生成后使用。

这条比赛实验路径保留 `/workspace` 目录约束。下方命令应在有相应挂载、转换工具和获准数据的环境运行，不能照搬为 macOS 上的基础示例。生成目录必须是新的目录：

```bash
python scripts/prepare_multitask_manifest.py \
  --sft /workspace/data/authorized-sft.jsonl \
  --reports-root /workspace/data/reports \
  --output-dir /workspace/experiments/new-report-manifest \
  --year 2018
```

有核验后的桥梁实体登记表时，追加 `--bridge-registry /workspace/data/bridge-registry.json --strict-entity-split`。登记表结构如下，哈希、实体及证据必须来自自己的审查，不要复制占位值作为正式证据：

```json
{
  "schema_version": 1,
  "evidence": "自己的实体核验记录",
  "reports": {
    "<原始报告的 SHA-256>": {
      "bridge_entity": "reviewed-bridge-id",
      "split": "development-regression",
      "evidence": "该报告对应实体的核验依据"
    }
  }
}
```

程序先校验全表中同一实体没有被分到不同 split，再允许 `--split` 筛选。实体互斥仍不等于独立盲测。`--diagnostic-only` 只选少量长度代表报告；清单继续标为开发回归，不能宣称泛化提升。当前清单的非诊断 `scope` 文字固定写 2018，即使传入其他 `--year`；使用其他年份时应以实际源记录核对范围，不能把这段文字当作审计证据。

为自己的部署填写 `configs/contest/ppu-native-lora.json` 的副本：`base`、`tokenizer`、`adapter` 指向实际本地资产，`served_model` 与服务暴露的名称一致。这个文件是模板，仓库不提供权重或已验收身份。仅填写 API URL 不够，运行器还会验证模型清单。

启动模型服务属于部署阶段。已有服务准备好后，可执行一次明确关闭 RSI 的分块基线：

```bash
python scripts/evaluate_multitask.py \
  --manifest /workspace/experiments/new-report-manifest/manifest.json \
  --model-manifest /workspace/config/my-model.json \
  --candidate-config configs/contest/chunked-baseline.json \
  --output-dir /workspace/experiments/new-chunked-baseline \
  --hardware my-recorded-hardware \
  --url http://127.0.0.1:8000/v1 \
  --backend vllm-api --engine direct --warmup 1 \
  --report-workers 1 --rsi-mode off
```

`chunked-baseline.json` 当前为 8,192 输入预算、4,096 输出预算、批量 1、种子 17。`whole-baseline.json` 保留整份路径作为另一对照；更换路径必须在结果中明示。`--backend sglang-api` 只是兼容 API 选项，不证明目标厂商已支持 SGLang。`--engine tongagents` 需要与当前代码和服务匹配的 SDK 验收；并行 SDK 另需 `--sdk-parallel-acceptance`。不能用 `direct` 的记录证明 SDK 链路通过。

`--report-workers 2/4` 控制报告并行，不等同于模型任务批量。批量、引擎参数、图执行或量化是否更快都需要单独测量；非空 `engine_parameters` 不能只改客户端配置就视为服务已应用。默认 `--visual-evidence off`；报告内含未解析图像时不能仅凭文件绑定认定完整覆盖。启用 `same-model` 需要相应视觉能力证据，不会自动生成该证据。

单独运行 `probe_fixed_model_vision.py` / `probe_report_visual_evidence.py` 前，还需显式准备 `CITYAGENT_BUDGET_LEDGER`，指向**已经存在的累计预算账本**，并与 CLI 的 `--budget-ledger` 一致；不得新建空账本来重置已消耗预算。固定原生 LoRA 策略另外绑定 `CITYAGENT_FIXED_MODEL_ALIAS`、`CITYAGENT_FIXED_ADAPTER_SHA256`，报告视觉探测绑定 `CITYAGENT_APPROVED_SOURCE_ROOT`。未配置值不可用于运行。固定策略保留原来的固定数据范围；自有数据应显式使用 `--model-policy manifest-locked`，并提供经批准的模型清单、来源清单哈希、`--source-root` 与 `--expected-denominator`，不能仅修改环境变量来扩大历史授权范围。

最终 `evaluation.json` 同时保存原始与最终三表指标、失败和固定分母，`records/` 与计时账本绑定真实执行。每份报告只计一次最终指标，不能按任务数摊薄错误。中断后可用以下工具重算完整分母：

```bash
python scripts/summarize_multitask.py \
  --manifest /workspace/experiments/new-report-manifest/manifest.json \
  --run-dir /workspace/experiments/new-chunked-baseline
```

它写入 `recomputed-evaluation.json`，校验已有产物并把缺失结果计为失败；不是恢复推理，也不会补跑模型。仅在旧执行器确认结束、预算任务已完成时，才可成对提供 `--budget-ledger` 与 `--job-id` 恢复对应计时。

## 时间与恢复指标的解释

- `timing.generation_seconds` 按模型调用账本统计；并行调用的累计秒数不能代替整批墙钟。输出 `infer_time.json` 的该数值也不是端到端总耗时。
- `total_wall_seconds` / `wall_seconds` 覆盖评测运行器记录的范围，`postwarm_batch_wall_seconds` 单独给出预热后的批次范围。外部启动模型服务、准备模型和导出权重仍需另记。
- 兼容字段 `latency` 只表示 `process_document`，不包括准备与排队。`end_to_end_latency_seconds` 从准备任务提交到报告结束，包含失败报告；应同时检查其 count 与 denominator。
- 恢复要测固定故障下的启动时间和恢复后完成时间，不能用“异常被捕获”或合成测试耗时代替。正常完成的一次运行也不能证明恢复不退步。

当前 RSI 验收器仍校验与实际报告收据一致的兼容延迟字段。新增端到端字段是额外观测，不能未经协议更新就当作新的 RSI 正式验收凭据。

## 固定权重推理 RSI

RSI 不修改底座、LoRA 或 tokenizer，不训练，也不引入 Judge。它优化可审计的推理动作、候选生成和有限搜索代码；正式推理默认关闭。入口能力分为三层：

| 工具 | 作用与限制 |
|---|---|
| `run_meta_optimization.py` | 最多八个配置候选的开发筛选；即使筛选较好也只输出 provisional 配置，冻结配置继续保留基线，`release_qualified=false` |
| `propose_rsi_strategy.py` | 在明确空闲窗口用模型生成绑定父版本的策略补丁，再做隔离回放与有限影子推理；会发出模型请求，不能在前台单卡任务活跃时随意启动 |
| `run_rsi_evolution.py` | 固定种子 17、受代数与唯一候选数上限约束的搜索、完整评测、恢复复测和发布核验；沿用同一个累计预算账本 |

先阅读帮助，核对自己的依赖与证据：

```bash
python scripts/run_meta_optimization.py --help
python scripts/propose_rsi_strategy.py --help
python scripts/run_rsi_evolution.py --help
python scripts/benchmark_rsi_recovery.py --help
```

递归入口要求显式的 `--manifest`、`--model-manifest`、`--baseline-config`、`--sandbox-root`、`--budget-ledger`、`--output-dir` 和 `--hardware`。输出、隔离根和预算账本保留 `/workspace` 限制；隔离必须真实通过，不支持时拒绝候选，不回退主进程执行。该入口默认 `--engine tongagents`，需要 SDK 证据；仅研究直接调用时显式选 `direct`，不能因此取得 SDK 验收。基线必须显式使用 chunked 或 bounded-react，whole 配置会被拒绝。

`run.py` 保留 `--rsi-mode off|frozen|experimental-online`；正式候选评测器只接受 `off|frozen`。混合策略在线批次、回放通过、自填 `verified/repeated`、跨试验缓存都不能替代独立真实收据。

晋级需要同版本执行器和相同条件下的两次独立完整评测：三表最终 F1 都不低于基线，失败、遗漏和冲突不增加，整批墙钟至少降低 20%，P50/P95 以及固定故障恢复启动与完成时间均不退步。预算不足或任一门槛未满足时保留基线。实验记录不是官方比赛成绩；当前公开 CI 不会为任何候选签发硬件发布资格。

## 不可直接混用的数据协议

1. **年份协议与实体协议不同。** `benchmark_lora.py` / `train_lora_nvidia.py` 读取 `meta.year`，固定把 2018 年留出。`prepare_training_revisions.py` 先按核验后的 `bridge_entity` 分组，输出子任务的 `bridge_entity/split`，没有现代 trainer 所需的 `meta.year`。直接传入会被拒绝；简单补上年份又会丢失实体划分约束。当前没有已接通的协议转换，不能描述为修订数据可无缝进入现代重训链路。
2. **document student 不是 LoRA。** `train_document_student.py` 拟合候选可靠性，按 document ID 哈希划分候选，并使用全部候选拟合最终导出模型。`holdout_raw_accuracy` 统计原始候选是否正确，不是导出模型在独立集上的预测准确率。自动标签匹配含模糊规则，需要人工确认；这些统计不能证明独立泛化。
3. **开发集回归不是盲测。** 同一报告的子任务、同一桥梁的跨年报告和经反复调优的数据都可能相关。划分必须先于分块；修订与裁定证据单独版本化，原标签不改写，生成端不接收标签或评分差异。

## Kev、NER 和硬件实验的公开使用边界

`experiments/kev_direct/` 保留候选选择、逐字回填、审核交接和运行探针。Kev 路径需要单独的 Kev 代码与 checkpoint，`infer.py` 还要求本地 `model-download.json` 及对应目录布局；它不是基础安装提供的第二个模型。`runtime.py` 的 CPU embedding / GPU decoder 是共享 NVIDIA 显存的容量策略，其 CUDA Graph 路径也是特定 CUDA 实验，不能作为 PPU 性能结论。

两个历史协调器 `single_review_runner.py`、`three_doc_runner.py` 使用 `deployment.py` 统一读取显式部署配置。变量均带 `CITYAGENT_KEV_` 前缀：

| 后缀 | 用途 |
|---|---|
| `EXPECTED_HOSTNAME`、`GPU` | 操作者选择的主机和 CUDA 设备 |
| `SDK_PYTHON`、`INFERENCE_PYTHON` | 分离的 SDK 与模型服务解释器 |
| `MAIN_MODEL_PATH`、`TOKENIZER_PATH` | 实际本地模型目录与 tokenizer 文件 |
| `LAUNCH_CONFIG` | 显式服务启动 argv 清单 |
| `MODEL_MANIFEST`、`MODEL_VERIFICATION`、`MODEL_VERIFICATION_SHA256` | 模型清单、`prepare_model_identity.py` 生成的准备收据及其固定 SHA-256 |
| `MAIN_PORT`、`MAIN_MODEL`、`SUMMARY_PORT`、`SUMMARY_MODEL` | 两个不同 loopback 端口和各自真实 served-model 名称 |
| `INPUT_ROOT`、`DOCUMENT_IDS`、`KEV_ROOT`、`KEV_PYTHON` | 两阶段 runner 额外需要的输入、逗号分隔文档 ID、Kev 根目录及解释器 |

这些配置没有可直接使用的私人默认值。路径必须存在且类型正确；launch 前缀必须严格为所选解释器、`-m`、`vllm.entrypoints.openai.api_server`，且唯一的 `--host` 明确设置为 `127.0.0.1`，模型目录也必须匹配。它不是可以填写任意命令的快速开始文件。`merge.py` 的 `--main-port --main-model --summary-port --summary-model` 全部必填，batch 配置也必须提供两个服务的端口与名称。

协调器保存 `kev-deployment-v1` 身份，绑定模型清单、准备收据、实际启动参数与 tokenizer 内容。恢复时必须一致；缺少该身份的旧实验目录不能直接复用。快速身份验证要求模型及 tokenizer 位于内核确认的只读挂载，并校验收据、文件库存和元数据；可变挂载或空的“已验证”文件会被拒绝，不能通过删除校验绕过。此限制避免恢复时重复读取数十 GiB 权重，但仍需在目标 Linux 环境验证真实恢复行为。

协调器仍依赖 Linux `/proc`、root 权限、NVIDIA 设备状态、TongAgents 2.7.20 和 `max_model_len=262144` 的服务条件，继续执行空闲、进程归属和服务身份检查。它们不是跨平台通用启动器。后续可将特定机器的生命周期编排移到独立部署适配层或历史归档，保留候选准备、交接格式和纯函数测试；不能为方便运行而删除现有所有权检查。本轮文档未执行这些服务管理脚本。

`experiments/ner_assisted/infer_index.py` 接收 `--manifest --checkpoint --output`，`--device` 默认 `cuda:0`。它加载已有 Ernie/UIE checkpoint 做推理，不训练；需要 `torch`、`transformers` 和 `safetensors`。manifest 应含带 ID 的原文 units，输出索引绑定来源及模型哈希，跨原文单元的预测会被丢弃。少量字段命中不能证明报告完整覆盖，索引也不应直接替代三表最终评分。

PPU kernel/runtime 探针、量化导出和 CUDA 推理应分别保留依赖指纹、加载记录、完整性及计时范围。硬件不同、共享资源状态不同或模型精度不同，结果就不能直接横比。具体格式和已存在的固定模型布局限制见 [量化文档](quantization.md)。

## 记录和发布结果

实验记录至少应包含源码版本、数据及模型身份、完整配置、依赖、硬件、真实调用、失败、中断、预热、缓存和计时范围。公开结果前检查产物中的源文、文件名、路径、凭据和部署信息，真实数据不得随代码自动上传。

公开仓库的 CPU CI 检查离线示例、测试和打包安装；不提供真实模型生成、训练、PPU、完整质量或恢复速度的验证结果。旧实验描述和历史分数只能按原来的来源、数据及协议解释，不能写成本次发布重新验证的成绩。
