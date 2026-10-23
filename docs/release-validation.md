# 0.2.0 发布验证

验证日期：2026-10-24。本地为 macOS、Python 3.11.15。基础开发环境使用仓库 `uv.lock` 与 `dev` extra；张量验证使用单独的临时 CPU 环境，不安装到厂商 PPU 环境。以下是本次实际执行结果。

| 检查 | 结果 |
|---|---|
| 完整 CPU 回归：`uv run --locked --extra dev python -m pytest -q` | **5,506 passed，22 skipped，608 subtests passed** |
| 独立张量环境：量化导出、LoRA 映射、训练工具合成测试 | **30 passed，14 subtests passed**；torch 2.14.1、safetensors 0.8.0、NumPy 2.4.6 |
| Kev 显式部署及恢复身份回归 | **95 passed**；未启动真实服务 |
| Ruff：`E9,F63,F7,F82` 与 diff 空白检查 | 通过 |
| 开发依赖兼容性及 pip-audit | 83 个已安装包兼容，未发现已知漏洞 |
| 独立 wheel 安装依赖 pip-audit | 未发现已知漏洞 |
| 源码树凭据扫描与隐私检查 | Gitleaks 通过；移除真实样本特例；人工复核公开源码、配置与展示图片 |
| 虚构 DOCX 示例与独立 wheel 验证 | 校验通过、生成三张 Word 表与证据 JSON，模型调用为零；在源码目录外读取包内配置/提示词成功 |
| 文档与展示素材 | 本地链接检查通过；仅放行 5 张经审阅的 PNG，EXIF 仅含尺寸；没有 PPT、备注或嵌入工作簿 |
| 无张量框架的 CLI 帮助 | 训练、token 统计、合并和量化入口通过 |

22 个跳过项涉及可选张量/模型资产、未发布的开发语料或平台能力；其中本次新增的张量数值检查已在独立环境补测。假后端用例的 tokenizer 回退警告不代表真实模型等价性验证。首次全套运行中的旧路径断言已修正后，重新执行以上完整回归。

这些检查没有加载真实模型、执行训练、访问 PPU 或进行比赛提交；没有重新测量答辩页中的训练时长、准确率或吞吐。RSI 仍默认关闭，真实质量、速度及恢复发布门槛需独立验证。训练依赖清单是可选 API 约束，不是经过所有硬件验证的环境锁。

GitHub 上最终发布提交的 Linux 测试与凭据扫描结果，见 [Tests](https://github.com/eust-w/cityagent-public/actions/workflows/tests.yml) 和 [Secret scan](https://github.com/eust-w/cityagent-public/actions/workflows/secrets.yml)。代码、源码交付包和依赖安装验证不能替代真实模型训练、完整权重合并与目标硬件验收。

公开历史已重新整理，旧版本记录由 0.2.0 替代。此前缓存清理申请尚未获得清除确认；Git 推送不能删除第三方克隆或保证 GitHub 的旧提交缓存立即消失。以下保留 0.1.0 的历史验证范围，不能将其数量当作本次测试结果。

---

# 0.1.0 发布验证

公开历史重建日期：2026-10-23。运行时代码和依赖未因历史重建而更改；以下本地回归于 2026-10-22 在同一份运行时代码上完成。环境为 macOS、Python 3.11.15、独立虚拟环境，依赖由本仓库 `uv.lock` 和 `dev` extra 安装。重建后提交的 Linux 检查以本文下方的 GitHub Actions 记录为准。

| 检查 | 结果 |
|---|---|
| 完整 CPU 回归：`python -m pytest -q` | 5,415 passed，20 skipped，596 subtests passed |
| Ruff：`E9,F63,F7,F82` | 通过 |
| Python 源码编译检查 | 通过 |
| `git diff --cached --check` | 通过 |
| 锁定的开发依赖兼容性检查 | 通过 |
| 当前锁定开发依赖的 pip-audit / OSV 查询 | 未发现已知漏洞 |
| 发布文件 Gitleaks 扫描 | 未发现凭据；仅对白名单中的三个已核实 tokenizer 资产 SHA-256 排除误报 |
| wheel 安装到独立环境、在源码目录外运行 | 包内配置和提示词读取成功；虚构示例生成三张 Word 表和证据 JSON，模型调用为零 |
| wheel 与源码交付包许可证 | 包含 LICENSE 和 NOTICE |
| 独立代码复审 | 缓存身份、许可证交付、公开默认配置和凭据扫描配置复审通过 |

跳过项涉及未提供的可选 tokenizer 资产或平台能力。部分测试使用受控假后端，会产生 tokenizer 回退警告；这些测试不代表真实 tokenizer 或模型等价性验收。测试夹具中的报告和回执均用于软件回归，不能作为比赛成绩或硬件性能证据。

Linux CI 单独执行完整测试、离线示例和 wheel 安装检查，具体提交的结果见 [GitHub Actions](https://github.com/eust-w/cityagent-public/actions/workflows/tests.yml)。Linux worker 内存测试使用当前执行映像的 `VmHWM`，避免继承父进程的历史峰值；256 MiB 上限保持不变。

这次发布没有运行云端模型、训练或比赛提交。RSI 仍默认关闭，须通过完整质量、速度与恢复门槛后单独启用。扫描与依赖审计是此时点、此发布范围的检查，不保证未来不存在漏洞。
