# Contributing to BERT-LLM-Yuri-cls

首先，感谢你对本项目的关注！🎉

> **📢重要声明**
> 由于当前维护者（@ArtistC220）面临学业和工作压力，**个人时间非常有限，可能无法及时响应 Issues 和 PR**。如果你对本项目感兴趣，**非常欢迎你接手维护或成为核心贡献者**！这不会给我造成任何负担，相反我会非常感激。请直接通过邮件或 GitHub Issues 联系。

## 项目当前状态与急需帮助的方向

本项目目前处于**维护模式**，以下是我们最急需帮助的方向（按优先级排序）：

| 优先级 | 方向                     | 说明                                                        | 难度 |
| :----: | ------------------------ | ----------------------------------------------------------- | :--: |
|   高   | **Bug 修复**       | 现有脚本的稳定性问题                                        |  高  |
|   高   | **推理脚本优化**   | 优化推理脚本的易用性               |  中  |
|   低   | **训练脚本模块化** | 将 `model_BERT_train*.py` 重构为可配置的训练模块          |  高  |
|   低   | **更多预训练模型** | 支持其他中文 BERT 变体（如 MacBERT、BERT-wwm-ext-large 等） |  中  |
|  未来  | **多模态支持**     | 扩展至百合漫画、动画的轻重分类                              |  高  |

**如果你愿意接手项目维护**，请直接联系：

- 邮件：ArtistC220@outlook.com（标题以 `[接手维护]` 开头）
-  GitHub：直接创建 Issue 表明意愿

## 开发环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/ArtistC220/BERT-LLM-Yuri-cls.git
cd BERT-LLM-Yuri-cls

# 2. 创建虚拟环境
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型（二选一）
# 方式A：轻量版（小模型）
# 从 Releases 下载并解压到 models/checkpoint-47200
# 方式B：large版（效果更好）
# 从 HuggingFace 下载 chinese-roberta-wwm-ext-large 微调版

# 5. 配置 API Key
# 编辑 config.json，填入你的 Moonshot API Key
```

## 项目结构速览

```
├── script/              # 核心脚本目录
│   ├── clean_txt.py            # 文本清洗
│   ├── model_BERT_infer.py     # BERT 推理（P值）
│   ├── dialogue_cut.py         # 台词切分
│   ├── LLM_dialogue.py         # LLM 分析台词（D值）
│   ├── LLM_verb.py             # LLM 分析动作/心理（A值）
│   ├── *_normalizer.New.py     # 数据归一化
│   ├── weighted_fun.py         # 加权融合（0.6P+0.3D+0.1A）
│   ├── *_merge.py              # 结果合并  需完善
│   └── model_BERT_train*.py    # 训练脚本（较原始）
├── config.json          # 全局配置（路径、API Key、参数）
├── run_all.py           # 主入口（现支持断点续跑）
└── txt_test/            # 放置待分析的文本文件
```

**关键配置文件**：`config.json`

- `bert_checkpoint`: 模型路径
- `api_key`: Moonshot API Key
- `LLM_threads_*`: LLM 并发线程数

## 如何报告问题（Reporting Issues）

请通过 [GitHub Issues](https://github.com/ArtistC220/BERT-LLM-Yuri-cls/issues) 提交问题，并选择对应模板：

### Bug 报告模板

```markdown
**问题描述**
清晰描述遇到的 bug

**复现步骤**
1. 运行 '...'
2. 输入 '...'
3. 出现错误

**预期行为**
你期望发生什么

**实际行为**
实际发生了什么

**环境信息**
- OS: [e.g. Windows 11]
- Python: [e.g. 3.10]
- GPU: [e.g. RTX 3060]

**错误日志**
```

粘贴错误信息

```

**附加信息**
截图或其他上下文
```

### 功能请求模板

```markdown
**功能描述**
你希望添加什么功能

**动机**
为什么需要这个功能

**实现思路**（可选）
如果你知道怎么实现，请描述
```

## 提交 Pull Request 流程

### 1. Fork 与分支

```bash
# Fork 本仓库后克隆你的 Fork
git clone https://github.com/YOUR_USERNAME/BERT-LLM-Yuri-cls.git
cd BERT-LLM-Yuri-cls

# 创建功能分支
git checkout -b feature/你的功能名称
# 或修复分支
git checkout -b fix/修复的bug描述
```

### 2. 开发与测试

- **代码风格**：遵循 PEP 8
- **配置文件**：如需修改默认路径，更新 `config.json` 并在 PR 中说明
- **测试**：在本地运行 `run_all.py` 确保全流程通过

### 3. 提交规范

```bash
# 提交信息格式
git commit -m "类型: 简短描述"

# 类型包括：
# feat: 新功能
# fix: 修复 bug
# docs: 文档更新
# refactor: 重构
# perf: 性能优化
# chore: 杂项

# 示例：
git commit -m "fix: 修复 LLM_dialogue.py 在空文本时的崩溃问题"
git commit -m "feat: 添加对 MacBERT 模型的支持"
```

### 4. 创建 PR

在 PR 描述中包含：

- **改动摘要**：一句话描述
- **详细说明**：改了什么，为什么改
- **测试结果**：测试了哪些场景
- **关联 Issue**： Fixes #123（如有）

## 给新贡献者的建议

### 入门级任务（Good First Issues）

如果你是第一次参与，可以从以下简单任务开始：

1. **完善错误处理**：为某个脚本添加更友好的错误提示
2. **改进日志输出**：让进度显示更清晰
3. **文档修正**：修复 README 中的错别字或过时信息
4. **添加注释**：为复杂函数添加中文/英文注释

### 中级任务

1. **断点机制完善**：改进 `run_all.py` 支持单步骤调试
2. **配置文件验证**：添加 `config.json` 的合法性检查
3. **Prompt 优化**：改进 LLM 判定准确率（需要测试数据）

### 高级任务

1. **训练脚本重构**：将 `model_BERT_train03.py` 模块化
2. **支持新模型**：添加对其他中文预训练模型的支持
3. **性能优化**：优化推理速度或内存占用

## 行为准则与许可证

### 行为准则

- 尊重所有参与者，保持友善和建设性
- 接受建设性批评，专注于技术讨论
- 避免使用歧视性、骚扰性语言

### 许可证

所有贡献将遵循 [MIT 许可证](LICENSE)。你提交代码即表示同意将代码授权给本项目使用。

## 联系方式与响应预期

| 方式          | 联系                                                                  |           预期响应时间           |
| ------------- | --------------------------------------------------------------------- | :------------------------------: |
| GitHub Issues | [新建 Issue](https://github.com/ArtistC220/BERT-LLM-Yuri-cls/issues/new) | **较慢**（维护者时间有限） |
| 邮件          | ArtistC220@outlook.com（标题以 `[BERT-Yuri-CLS]` 开头）             |          **较慢**          |
| 公众号        | MathArtistC（紧急事务）                                               |               较快               |

> ⚠️ **关于响应时间的说明**：由于维护者个人原因，Issues 和 PR 可能不会及时处理。**如果你愿意帮助 review PR 或维护项目，请一定联系我！**

---

## 致谢

感谢所有为本项目做出贡献的人！无论是一个小小的 typo 修复，还是重要的功能开发，你们的贡献都让这个项目变得更好。

**特别期待**：如果你有兴趣长期维护本项目，请毫不犹豫的联系我，我会很乐意将项目交接给有热情的新维护者。

---

*最后更新：2026年2月*
