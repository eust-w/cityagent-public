# 模型合并与实验量化

`experiments/prepare_ppu_quantization.py` 是针对指定模型布局的 **CPU 权重导出工具**：先把 LoRA 合并为 BF16，再从该 BF16 产物导出 W8A8 INT8。它不读取数据集、不训练，也不启动模型服务。成功导出不代表推理引擎能够加载，不代表质量、速度或恢复能力已通过验收。

源码保留了原有严格布局检查和量化计算。本页中的模型、adapter 和输出路径都是需要替换的示例；仓库不包含这些资产。

## 环境边界

| 环境 | 可以做什么 | 额外要求 |
|---|---|---|
| 公共 CPU 开发环境 | 查看帮助、运行不依赖张量框架的测试；另建 CPU 导出环境后执行合并与量化 | 配置兼容的 `torch`、`safetensors` 和 NumPy；基础安装和 `uv.lock` 不包含这些导出环境依赖 |
| NVIDIA 训练环境 | 显式执行 `scripts/train_lora_nvidia.py` 等训练工具 | 独立配置 CUDA、训练框架和符合脚本要求的 GPU；训练不是量化导出的前置自动动作 |
| PPU 推理环境 | 使用目标厂商支持的运行时验证导出模型 | 保留厂商提供的 torch/vLLM；先验证该运行时是否支持输出格式，不用通用 pip 包覆盖厂商依赖 |

查看帮助只需要 Python 标准库：

```bash
python experiments/prepare_ppu_quantization.py --help
```

实际导出请准备独立 Python 环境，例如 `.venv-export`，先安装适合操作系统的 CPU PyTorch，再安装 `safetensors`、NumPy 和测试工具：

```bash
.venv-export/bin/python -m pip install safetensors numpy pytest
.venv-export/bin/python -m pip check
```

NumPy 也是完整合成测试中 safetensors 张量写入所需的依赖，不能仅安装 torch/safetensors 就假设测试环境齐全。使用该环境的 Python 运行下方命令。缺少 torch/safetensors 时工具会报出明确错误并停止，不会自动安装依赖或创建模型输出目录。

基础仓库 CI 验证帮助与缺依赖行为；若环境没有 torch/safetensors，张量计算测试会明确跳过。不能把这一结果描述为完整量化测试已经通过。

## 支持范围

以下是现有代码的硬性条件，不是通用模型推荐：

| 项目 | 要求 |
|---|---|
| 完整架构 | `Qwen3_5ForConditionalGeneration`，保留视觉塔 |
| 文本配置 | 64 层，`hidden_size=5120` |
| 输入底座 | 未量化 BF16；具有 `config.json`、`model.safetensors.index.json` 和完整权重分片 |
| LoRA | 普通 `LORA`，`r=16`、`lora_alpha=32`、`bias=none`，F32 adapter 张量 |
| LoRA 覆盖 | 指定语言层的 256 个目标矩阵，A/B 配对完整、形状与底座一致 |
| 量化覆盖 | 精确匹配源码中的 400 个矩阵；其他张量保留 |
| 不支持的 adapter 选项 | DoRA、rsLoRA、`fan_in_fan_out`、可变 rank/alpha、`modules_to_save`、层复制等 |

具体匹配规则见脚本中的 `LORA_TARGET`、`QUANT_TARGET`、`inspect_source()` 和 `inspect_adapter()`。输入名称相似不代表符合布局；不应通过删除这些检查来适配另一底座。9B 模型、其他层数或其他 LoRA 配置不能直接使用此工具。

## 导出流程

输出目录必须尚不存在，位于输入目录之外，且 BF16 与 INT8 输出目录不能互相嵌套。先准备足够磁盘空间：两个完整导出会同时占用空间，源权重和 adapter 也继续保留。

仅合并到 BF16：

```bash
.venv-export/bin/python experiments/prepare_ppu_quantization.py \
  --base /path/to/compatible-bf16-base \
  --adapter /path/to/compatible-lora \
  --bf16-output /path/to/new-merged-bf16 \
  --bf16-only
```

一次完成 BF16 合并与 INT8 导出：

```bash
.venv-export/bin/python experiments/prepare_ppu_quantization.py \
  --base /path/to/compatible-bf16-base \
  --adapter /path/to/compatible-lora \
  --bf16-output /path/to/new-merged-bf16 \
  --int8-output /path/to/new-w8a8 \
  --threads 2 --row-chunk 512 --shard-bytes 1000000000
```

也可以从**本工具已完成并通过文件哈希检查的 BF16 导出**继续量化：

```bash
.venv-export/bin/python experiments/prepare_ppu_quantization.py \
  --from-merged /path/to/completed-merged-bf16 \
  --int8-output /path/to/new-w8a8
```

`--from-merged` 要求来源包含本工具生成的 `conversion_manifest.json`，状态为 `completed`、模式为 `merged_bf16`，且全部 256 个 LoRA 目标已合并、产物哈希匹配。任意外部 BF16 权重目录不能作为已经验收的合并结果传入。

`--threads` 控制 CPU 并行，`--row-chunk` 控制矩阵逐行计算的分块大小，`--shard-bytes` 控制输出分片目标大小。单个张量可能大于分片目标；这些参数不保证固定的峰值内存或更高速度。

## 输出与计算语义

- 合并计算为 `W + 2 × (B @ A)`，其中 2 来自此工具强制的 `alpha/r = 32/16`，之后写出 BF16。
- 权重量化为对称、逐输出通道 INT8；激活方案标记为动态、逐 token A8。导出不使用校准数据。
- 输出包含 safetensors 分片、索引、辅助模型配置、逐矩阵统计和 `conversion_manifest.json`。INT8 配置使用 `quant_method=w8a8_int8`，不是自动转换为任意引擎通用的 compressed-tensors、AWQ 或 GPTQ 格式。
- 来源文件不被覆盖。失败保留 `partial` 状态和已完成产物记录；未完成目录不能用于推理，也不能作为 `--from-merged` 的输入。
- `ppu_runtime_validated` 与 `quality_validated` 保持 `false`。导出器不能自行签发运行时或质量验收。

合并 BF16 与原生动态 LoRA 在浮点舍入和执行路径上不同，需要分别比较，不能直接沿用原生 LoRA 的性能与质量记录。

## 与其他训练和合并脚本的关系

`scripts/train_lora_nvidia.py` 保留完整多模态模型，并在 `model.language_model.layers.*` 上训练 adapter。其参数可选择不同 rank/alpha；这不意味着所有导出都符合上面固定的 27B 布局约束。

`scripts/merge_lora_v3.py` 是另一条合并路径，现已显式支持两种完整 PEFT 键名前缀：旧文本 adapter 的 `base_model.model.model.layers.*`，以及完整多模态 trainer 的 `base_model.model.model.language_model.layers.*`。同一 adapter 必须只用其中一种布局；可合并模块限于 `self_attn` 的 q/k/v/o 投影及 MLP 的 gate/up/down 投影。程序在修改底座前检查全部映射，拒绝未知键、重复目标、混合前缀、缺失 A/B、rank/形状不符和底座中不存在的目标。

这修复了新 trainer 与旧合并入口之间的命名差异，**不代表任意 adapter 与任意底座都兼容**，也不放宽本页量化导出器的 64 层、5120 维和 r16/alpha32 条件。`merge_lora_v3.py` 必须显式传入 `--base-model`、`--adapter` 和新 `--output`，本地模型与 tokenizer 加载禁用远程自定义代码；实际合并还检查 adapter、FP32 增量和最终参数的有限性，包括 BF16 溢出。其 `--check` 只检查路径和配置，不加载张量；双前缀映射和小张量测试不能替代真实权重合并、重载和生成质量验证。

## 最小验证

在公共开发环境运行无需模型的入口测试：

```bash
python experiments/prepare_ppu_quantization.py --help
python -m pytest -q tests/test_prepare_ppu_quantization.py \
  -k QuantizationCommandHelpTests
```

在另行准备的 CPU 导出测试环境中，运行完整的合成张量测试：

```bash
.venv-export/bin/python -m pytest -q tests/test_prepare_ppu_quantization.py
```

测试环境需安装 pytest、NumPy 及上述张量依赖。完整测试使用微小合成张量验证合并、量化、输出哈希与失败记录，仍不能替代真实模型加载或真实报告质量评测。

部署前应在目标推理引擎上完成真实加载、短生成、完整报告质量、处理速度和恢复速度比较。RSI 的发布条件另行验证；量化文件导出成功不会启用 RSI。

## 资产与许可

模型底座、adapter、训练数据、厂商运行时和第三方 SDK 需要自行取得合法使用权限。仓库 Apache-2.0 代码许可不覆盖这些外部资产，也不自动授予导出模型的再分发权。分享合并或量化权重前，应核对原模型、adapter 及相关依赖的许可与发布条件；不要把私人路径、凭据或运行记录随权重公开。
