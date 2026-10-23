# 数据准备与可选训练

这些工具用于用户自行准备的授权数据和本地模型。正常推理、离线示例和推理 RSI 不会自动调用训练。仓库不附带训练报告、标签、teacher 产物、模型权重或已验收的硬件环境。

| 工具 | 用途 | 执行边界 |
|---|---|---|
| `build_sft_dataset.py` | 报告与标签精确配对、生成整报告 SFT | 显式指定数据目录；可先 `--audit-only` |
| `token_stats.py` | 用本地 tokenizer 统计完整输入与答案长度 | 不加载模型权重，不估算训练速度 |
| `benchmark_lora.py` | Qwen3.5 LoRA 的 NVIDIA 容量和吞吐实验 | `--prepare-only` 才只准备；正常运行会执行反向传播，但不保存权重 |
| `train_lora_nvidia.py` | Qwen3.5 多卡 FSDP2 LoRA 训练 | `--prepare-only` 才只准备；正常运行会训练并保存 adapter |
| `train_lora.py` | 显式配置的兼容训练入口 | 默认只准备；必须增加 `--execute` 才训练 |
| `evaluate_generation_nvidia.py` | 冻结 SFT 上的真实生成与评分 | 会加载底座和 adapter 并调用 GPU |
| `prepare_training_revisions.py` | 带来源证据的标签修订和分块 SFT | 只准备数据；尚未接通现代 trainer 的划分协议 |

以下示例均从仓库根目录执行。`/path/to/...` 是占位路径，输出应使用新的目录。`--help` 可以检查当前参数，不会启动训练。

## 独立安装训练环境

使用 Python 3.11+，为训练单独建虚拟环境。不要在赛机的厂商 PPU torch/vLLM 环境内安装下面的依赖。

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
```

实际 NVIDIA 训练前，先按自己的驱动和 CUDA 环境安装兼容的 PyTorch，当前现代训练脚本要求 `torch>=2.9` 且支持 CUDA BF16。安装方式由目标机器决定；本仓库不提供通用 CUDA wheel 安装命令。

安装好 torch 后，把其实际版本写成约束，再安装其余依赖：

```bash
python -c 'import torch; print("torch==" + torch.__version__)' > .venv-training/torch-constraint.txt
python -m pip install -c .venv-training/torch-constraint.txt -r requirements-training-nvidia.txt
python -m pip install -c .venv-training/torch-constraint.txt -e .
python -m pip check
```

兼容入口使用 `requirements-training-legacy.txt`，替换上面的 requirements 文件即可。两份清单不直接安装 torch，但其他包会声明 torch 依赖，因此仍需保留约束。依赖冲突时应停止并检查环境，不应去掉约束来覆盖厂商版本。

这些是源码要求的最低 API 版本及主版本范围，不是经过所有硬件验证的锁文件。每次实验应保存解析后的依赖版本、模型身份和运行清单。QLoRA 还需要与 CUDA 匹配的 bitsandbytes；`flash_attention_2` 需要单独安装兼容的 flash-attn。下面的 FSDP2 示例使用 `sdpa`。

仅做 tokenizer 统计或 prepare-only 时可以使用独立 CPU 环境，并安装支持所选本地模型 tokenizer 的 Transformers；不需要 GPU 或模型权重加载。公开兼容入口尚未在 PPU 上重新验收，CPU 准备通过也不代表硬件训练通过。

## 整报告 SFT

`build_sft_dataset.py` 的输入是两个独立根目录，下面各有年份子目录。当前实现扫描 `2012年`、`2013年`、`2014年`、`2016年`、`2017年`、`2018年`；它不是自动发现任意年份的通用目录扫描器。标签需符合三表格式，`.doc` 转换需要 LibreOffice 的 `soffice`。

先审计配对，再用另一个新目录生成数据：

```bash
python scripts/build_sft_dataset.py \
  --train-root /path/to/authorized-reports \
  --label-root /path/to/authorized-labels \
  --out-dir /path/to/new-pairing-audit --audit-only

python scripts/build_sft_dataset.py \
  --train-root /path/to/authorized-reports \
  --label-root /path/to/authorized-labels \
  --out-dir /path/to/new-sft --workers 2
```

只接受唯一的一对一精确文件名配对；模糊候选留给人工复核。检查 `pairing_manifest.json`、`stats.json`、`anomalies.md` 和 `sft_train.jsonl`，不要只看命令退出状态。当前构建器会跳过读取失败、报告文本过短、标签解析失败或病害明细为空的记录，这些筛选使生成记录数不一定等于输入报告数。

现代 LoRA 工具要求每行有非空字符串 `instruction`、`input`、`output`，以及明确的 `meta.year`（例如 `"2017"` 或 `"2017年"`）。重复提示和歧义年份会被拒绝。输出字段是序列化后的目标 JSON 字符串；真实内容及源文件名、哈希、转换缓存都应保留在用户自己的数据目录。

当前 `benchmark_lora.py`、`train_lora_nvidia.py` 和生成评测固定将 **2018 年全部留出**，其他年份用于训练。这是内部年份留出协议，不保证同一桥梁跨年份互斥；不能据此宣称桥梁实体上的独立泛化，也不是官方测试成绩。

## 完整序列长度统计

```bash
python scripts/token_stats.py \
  --model /path/to/local-model \
  --sft /path/to/new-sft/sft_train.jsonl \
  --max-lengths 8192 16384 32768 \
  --output /path/to/new-token-stats.json
```

统计入口接受任意包含非空 `instruction/input/output` 的 JSONL，不要求年份。它使用本地 tokenizer 的聊天模板与真实 EOS，统计完整提示和答案；不会截去长报告中段，也不会把 token 数外推成某块卡的速度。`--output` 可省略，指定时使用新文件。

## NVIDIA 基准与训练

现代脚本只接受完整的 `Qwen3_5ForConditionalGeneration` 架构，不能自动当作其他底座的通用 trainer。tokenizer 和模型均从显式本地路径读取，不允许远程自定义代码。

先在单进程中准备清单：

```bash
python scripts/train_lora_nvidia.py \
  --model /path/to/local-model \
  --sft /path/to/new-sft/sft_train.jsonl \
  --out-dir /path/to/new-train-plan --prepare-only

python scripts/benchmark_lora.py \
  --model /path/to/local-model \
  --sft /path/to/new-sft/sft_train.jsonl \
  --out-dir /path/to/new-benchmark-plan \
  --mode bf16-sharded --length-bucket 8k --prepare-only
```

检查清单中的样本选择、留出集、完整序列长度、排除原因和哈希。超长行整条排除并留痕，不会进行首尾拼接；因此训练子集与完整评测分母不同。prepare-only 只加载 tokenizer，不分配模型权重；不能在 `torchrun` 多进程下执行这个模式。

下面两个命令会真正训练，需要先确认 GPU 容量和实验预算，并选择全新的输出目录。`--global-batch-size` 必须能被 GPU 进程数整除；示例的两卡和批量 8 只是参数示例，不是容量保证。

```bash
torchrun --standalone --nproc_per_node=2 scripts/benchmark_lora.py \
  --model /path/to/local-model \
  --sft /path/to/new-sft/sft_train.jsonl \
  --out-dir /path/to/new-benchmark-run \
  --mode bf16-sharded --length-bucket 8k \
  --max-steps 6 --warmup-steps 2 --global-batch-size 8

torchrun --standalone --nproc_per_node=2 scripts/train_lora_nvidia.py \
  --model /path/to/local-model \
  --sft /path/to/new-sft/sft_train.jsonl \
  --out-dir /path/to/new-train-run \
  --epochs 2 --global-batch-size 8 --max-length 32768
```

`benchmark_lora.py` 正常运行会更新临时模型参数，但不保存权重、不恢复 checkpoint。其吞吐指标只涵盖 GPU 同步后的优化步，不包含加载、tokenizer、步间日志和留出损失计算；不能当作端到端报告推理速度。留出损失用于数值检查，不能替代三表质量评测。QLoRA 是该基准的另一种模式，需显式 `--mode qlora`；它不能使用 `--activation-offload`。

`train_lora_nvidia.py` 默认每轮访问全部符合长度条件的训练样本一次，尾批填充的损失权重为零。它保存 LoRA adapter，例如 `epoch-001/`；提前停止则在目录名及摘要中标记 partial。它不自动续训、不恢复优化器状态、不覆盖已有输出。检查 `manifest.json`、`summary.json`、加载审计和 adapter 导出记录，随后实际重载并评测。

## 兼容入口：默认只准备

```bash
python scripts/train_lora.py \
  --model /path/to/local-model \
  --sft /path/to/new-sft/sft_train.jsonl \
  --out-dir /path/to/new-legacy-plan --prepare-only
```

`--model`、`--sft`、`--out-dir` 都必须显式填写。省略 `--execute` 仍然只做准备；准备会加载本地 tokenizer 并写清单，不读取训练权重。它拒绝输出复用，不会从目录里的旧 checkpoint 自动续训，也不会截断超长样本后训练。

| 参数 | 行为 |
|---|---|
| `--mode train\|smoke\|benchmark` | 选择训练或短实验，仍需要 `--execute` 才实际执行 |
| `--max-steps` | smoke/benchmark 必须显式提供 1–100；train 可选 |
| `--max-samples`、`--max-length` | 显式限制样本数及完整序列长度；长度默认 32768 |
| `--holdout-year` | 默认 2018；修改后不再与现代固定年份协议等同 |
| `--model-class` | `image-text-to-text`（默认）或 `causal-lm`；选定后不自动回退 |
| `--device`、`--seed` | 仅支持 `cuda:0`，随机种子默认 42 |
| `--use-liger-kernel` | 默认关闭；启用前需自行安装并验证对应实现 |

实际执行前通过 `CUDA_VISIBLE_DEVICES` 只暴露一张设备；脚本使用该可见设备的 `cuda:0`，不自动跨设备。默认 `image-text-to-text` 对应完整 `Qwen3_5ForConditionalGeneration`；`causal-lm` 仅接受本地配置明确声明的纯文本 `Qwen3_5ForCausalLM` 或 `Qwen3ForCausalLM`，不会把完整多模态模型自动降成文本模型。加载与权重键审计不完整时会拒绝训练。

明确执行一个限步烟雾实验的示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model /path/to/local-model \
  --sft /path/to/new-sft/sft_train.jsonl \
  --out-dir /path/to/new-legacy-smoke \
  --mode smoke --max-steps 1 --max-samples 2 --execute
```

这是模型加载与反向传播实验，会消耗硬件时间。`train` 和 `smoke` 成功后保存到 `<out-dir>/adapter/`；`benchmark` 只保存统计，不保存权重，其耗时仅代表实际观察到的训练循环，不外推整语料吞吐或总训练时间。非有限 loss 或 adapter 张量不能被标记为成功。当前严格权重键审计不支持 tied embeddings，会显式拒绝对应配置。兼容入口不重现旧的截断样本或历史分数，也不代表单卡 PPU 已通过训练验收。

## adapter 生成评测与合并检查

先从现代 prepare-only 清单冻结完整 2018 留出集合。token manifest 必须包含每一行的审计元数据，包括训练中被长度过滤的行；不能用选中的训练子集构造评测分母。

```bash
python scripts/prepare_generation_manifest.py \
  --sft /path/to/new-sft/sft_train.jsonl \
  --token-manifest /path/to/new-train-plan/manifest.json \
  --output /path/to/new-generation-selection.json

python scripts/evaluate_generation_nvidia.py \
  --model /path/to/local-model \
  --adapter /path/to/new-train-run/epoch-001 \
  --sft /path/to/new-sft/sft_train.jsonl \
  --selection-manifest /path/to/new-generation-selection.json \
  --partition diagnostic --out-dir /path/to/new-generation-diagnostic
```

第二个命令会实际生成。它绑定 SFT、清单、代码、模型配置、adapter 和 token IDs，重载 adapter 后核对张量，再进行评分。标签只用于 CPU 上的前缀一致性检查及生成后的评分，后端只接收提示 tokens。

这里示例的 `epoch-001/` 来自现代完整多模态 trainer；兼容入口的产物目录是 `adapter/`。此生成评测器固定加载 `Qwen3_5ForConditionalGeneration`，不能直接用于兼容入口 `causal-lm` 训练的其他架构。

诊断完成后，用新输出目录、相同冻结清单以及 `--partition final_acceptance` 运行对应集合。小数据集可能没有该分区，脚本会拒绝空集合。`--limit`、`--sample-sha256` 和分片用于明确标记的诊断/分片运行，不能把一个子集结果当完整留出评测。失败样本仍计入分母。这个入口仍是内部模型生成评测，不包括完整正式流水线的恢复与发布验收。

需要合并前可先检查路径与配置：

```bash
python scripts/merge_lora_v3.py \
  --base-model /path/to/local-model \
  --adapter /path/to/new-train-run/epoch-001 \
  --output /path/to/new-merged-model --check
```

`--check` 不加载权重，不能证明 adapter 张量、数值合并结果或生成质量正确。实际合并仅支持源码明确实现的权重命名和普通 LoRA 配置；adapter 必须对应同一个底座。使用新输出目录，完成合并后另行重载验证。

合并入口的三个路径参数均为必填，不使用历史部署目录；模型和 tokenizer 均禁止远程自定义代码。它支持旧文本 adapter 的 `base_model.model.model.layers.*` 与完整多模态 adapter 的 `base_model.model.model.language_model.layers.*` 两种布局；同一个 adapter 不可混用前缀。映射只允许文本 self-attention 的 q/k/v/o 投影与 MLP 的 gate/up/down 投影，拒绝未知键、重复目标、缺失 A/B、rank 或矩阵形状不匹配。

## teacher、文档 student 与修订子任务

teacher/标签工具是离线数据准备入口。先查看 `build_document_teacher_corpus.py`、`build_official_teacher_samples.py`、`align_teacher_to_official_labels.py` 的 `--help`，显式提供自己的源报告、标签、裁定记录和输出路径；不要把真实数据或历史样本特例写进源码。外部专家命令由用户显式配置，可能运行用户提供的程序，应与正式推理环境分开。

`train_document_student.py` 训练的是文档候选校准模型，不是 LLM LoRA。当前脚本按 document ID 留出部分候选，最终另用全部候选拟合导出模型；其 `holdout_raw_accuracy` 是原始候选正确率，不是导出模型在独立集上的预测准确率。现有自动标签匹配包含模糊规则，使用前需核验配对，不应据此宣称独立质量提升。

`prepare_training_revisions.py` 先按核验后的桥梁实体划分，再要求来源绑定的批准修订与完整责任区注释，原标签保持不变：

```bash
python scripts/prepare_training_revisions.py \
  --manifest /path/to/approved-source-manifest.json \
  --revision-dir /path/to/approved-revisions \
  --output-dir /path/to/new-revised-sft \
  --tokenizer /path/to/local-model
```

**该输出尚未直接接通 NVIDIA trainer。** 子任务行保存 `bridge_entity/split`，而现代 trainer 要求 `meta.year` 并固定按年份划分。仅补年份会丢失实体划分约束；必须先实现并测试显式协议转换、分组隔离和来源绑定。当前版本不能把这一流程描述为已完成的修订数据重训链路。

## 本地验证与结果解释

纯数据准备及协议测试可使用开发环境运行：

```bash
python -m pytest -q \
  tests/test_sft_pairing.py tests/test_sft_tables.py \
  tests/test_benchmark_lora.py tests/test_train_lora_nvidia.py \
  tests/test_training_revisions.py tests/test_generation_nvidia_runner.py
```

这些测试不等于真实训练或硬件验收。发布模型前应分别报告数据审计、实际训练、权重重载、完整生成质量与运行速度；启用推理 RSI 仍需完成 [使用说明中的独立发布条件](usage.md#评测与发布条件)，不能用训练损失或单次短实验替代。
