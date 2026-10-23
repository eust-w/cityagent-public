# 使用说明

## 安装与离线示例

使用 Python 3.11+，在源码目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/offline_demo.py --output-dir /tmp/cityagent-demo
```

示例使用虚构文本生成 DOCX，在本机完成离线处理。没有模型下载或远端推理步骤。请使用新的输出目录，以便区分不同次运行的产物。

只使用命令行工具时可安装 `python -m pip install -e .`。可选开发依赖和测试并不包含 GPU / PPU 推理环境。旧版 `.doc` 依赖系统转换工具；高级路径明确检查 `soffice`。首次接入转换器时应检查转换质量记录，而非仅确认输出文件存在。

## 单份与批量处理

```bash
bridge-evidence single \
  --input /path/to/report.docx \
  --output-dir ./outputs \
  --json-dir ./outputs-json \
  --llm-mode off

bridge-evidence run \
  --input-dir /path/to/reports \
  --output-dir ./outputs \
  --json-dir ./outputs-json \
  --llm-mode off \
  --workers 2
```

`single` 打印该报告的结构化摘要，`run` 打印处理数量、校验状态、调用统计及失败列表。Word 输出用于阅读，JSON 保存结构化状态，工作目录保存文档中间表示与事件。应同时检查 `validation` 和失败记录；文件已生成不表示所有字段或来源均已确认。

使用自己的配置时，`--config` 放在子命令之前：

```bash
bridge-evidence --config ./my-config.json single \
  --input /path/to/report.docx \
  --output-dir ./outputs \
  --llm-mode off
```

配置字段见 `src/bridge_evidence/config.py` 和 `configs/base.json`。默认缓存与工作文件位于 `.bridge_evidence/`，不要提交包含真实报告的缓存、日志和输出。

## 显式接入模型服务

离线流程与在线流程的结果需要分别评估。在线模式需自行提供授权可用的服务及模型，并确认所选工作流要求的文本、embedding、reranker 等接口实际可用。仅有聊天 API 不足以保证完整工作流运行。

从 `configs/base.json` 复制一份配置，在本地填写：

| 字段 | 内容 |
|---|---|
| `llm_mode` | 显式设为 `on` |
| `base_url`、`model`、`fast_model` | 自有文本服务地址与实际模型 ID |
| `task_models` | 任务到模型 ID 的映射；检查是否需要覆盖默认映射 |
| `vision_model`、`vision_critic_model` | 所选工作流实际支持的模型，未配置的能力用 `null` |
| `semantic_base_url`、`embedding_model`、`reranker_model` | 自有语义服务和实际模型 ID |
| `api_key_env` | `BRIDGE_EVIDENCE_API_KEY` |
| `semantic_api_key_env` | `BRIDGE_EVIDENCE_SEMANTIC_API_KEY` |

通过运行环境的秘密管理机制设置以上两个密钥变量，不把密钥写进 JSON、代码或命令示例。将 endpoint 从默认 loopback 地址改为远端地址前，确认报告内容可以发送到该服务。

```bash
bridge-evidence --config ./my-online-config.json single \
  --input /path/to/report.docx \
  --output-dir ./outputs-online \
  --json-dir ./outputs-online-json \
  --llm-mode on
```

可用命令与参数以当前版本为准：

```bash
bridge-evidence --help
bridge-evidence single --help
bridge-evidence run --help
```

## 高级 `run.py` 入口

这是需要额外部署的推理入口，不属于无需模型的离线示例。先阅读帮助：

```bash
python run.py --help
```

需要自行准备模型及 tokenizer，按所选方案准备可选 LoRA 或已合并权重，并核对 `--model-manifest`。`weight_kind: "base"` 明确表示无 adapter 的底座方案；它与带 LoRA 的结果不是同一基线。模型准备与正式推理分开，入口不会在权重缺失时自动完成合并。

| 参数 | 用途 |
|---|---|
| `--input-dir`、`--result-dir`、`--work-dir` | 显式指定报告、交付文件和执行记录目录 |
| `--model-manifest` | 绑定模型、tokenizer、可选 adapter / 合并权重 |
| `--backend` | `local-vllm`、`vllm-api` 或 `sglang-api` |
| `--backend-url` | API 后端地址；该高级后端限制为 loopback 服务 |
| `--engine` | `direct` 开发路径或需要单独安装和验收的 `tongagents` |
| `--workflow` | `whole`、`chunked`、`bounded-react` 等；查看帮助中的完整列表 |
| `--candidate-config` | 预算、批量和提示等候选配置 |
| `--resume` | 身份匹配后复用已完成结果 |
| `--check` | 检查输入、模型、依赖和转换工具，不运行生成 |
| `--rsi-mode` | 保持 `off`，直至相关发布验收完成 |

API 模式仍需本地模型资产和 tokenizer 身份校验；只填写 URL 不能跳过这些要求。本地 vLLM、厂商运行时和 TongAgents SDK 不随基础 Python 包安装。按目标平台提供的兼容环境安装，不应为运行示例而覆盖已有驱动或厂商依赖。高级路径保留 `/workspace` 目录约定，使用前检查自己的挂载与权限。

恢复时保留原工作目录和身份，确认旧进程退出后再使用 `--resume`。改变模型、代码、提示或预算后，应使用独立运行目录。不要通过删除错误记录或修改验收状态来让失败结果看起来成功。

## 评测与发布条件

离线规则代理评测示例：

```bash
bridge-evidence evaluate \
  --input-dir /path/to/development-reports \
  --label-dir /path/to/evaluation-labels \
  --output ./evaluation.json \
  --llm-mode off
```

这与 `scripts/evaluate_multitask.py` 的高级执行评测不是同一协议。标签只用于合法的离线评测或明确隔离的数据准备，不得流入待测生成请求。请使用有权处理的数据，按桥梁实体划分开发与独立评测集，并保留失败样本在分母中。

启用 RSI 前，应固定模型、LoRA、tokenizer、评分和样本集合，在相同执行器及硬件条件下重测基线，再进行两次独立完整候选评测。各表质量、失败数量、来源完整性、冲突数量、整批耗时、报告延迟和故障恢复都需要真实记录。任何必要项未通过，就继续使用基线。

仓库不附带已批准的模型服务、完整评测集或通用 RSI 晋级证明，不能从单元测试或离线示例推导真实模型的质量和加速结论。
