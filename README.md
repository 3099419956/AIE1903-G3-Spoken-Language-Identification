# AIE1903 G3 语音语种识别

本仓库用于 AIE1903 第三组（G3）的期中小组项目，任务是从一段语音录音中
识别说话者使用的语言。项目基线采用 Meta 的 MMS 语种识别模型
`facebook/mms-lid-126`。

> 当前状态：仓库初始框架。数据集、实验结果、评估结果和最终报告尚未完成。

## 项目目标

搭建并评估一套可复现的语音语种识别流程：

1. 加载并检查音频样本。
2. 使用基于 MMS 的模型完成基线预测。
3. 选择合适的分类指标评估模型表现。
4. 至少完成一项扩展或对照实验，例如音频预处理、领域适配、特征提取器
   替换或分类器改进。
5. 在小组报告中说明方法、结果、局限性及错误分析。

## 参考资料

- 论文：[Scaling Speech Technology to 1,000+ Languages](https://jmlr.org/papers/v25/23-1318.html)
- 文档：[Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- 模型：[facebook/mms-lid-126](https://huggingface.co/facebook/mms-lid-126)

## 仓库结构

```text
.
├── data/        # 数据集说明和本地数据占位目录
├── docs/        # 项目记录、报告和展示材料
├── notebooks/   # 探索性分析和可复现实验
├── outputs/     # 生成的预测结果和图表（不提交到 Git）
├── scripts/     # 训练、推理和评估脚本
└── src/         # 可复用的 Python 代码
```

## 环境配置

建议使用 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

模型文件可能较大。如果本地磁盘空间有限，可以把 Hugging Face 缓存放到
仓库外部：

```powershell
$env:HF_HOME = "D:\hf-cache"
```

## 小组协作

提交 Pull Request 前，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
不要将原始音频、模型权重、生成结果、账号凭据或个人隐私信息提交到 Git。

## 初始任务

- [ ] 确定数据集、语言范围以及训练/验证/测试划分。
- [ ] 复现 `facebook/mms-lid-126` 基线预测。
- [ ] 确定主要评估指标，并统计各语言的分类结果。
- [ ] 完成一项扩展实验或控制变量对照实验。
- [ ] 补充错误分析并记录方法局限性。
- [ ] 完成最终报告和课堂展示。
