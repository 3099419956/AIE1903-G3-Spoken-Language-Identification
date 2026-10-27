# AIE1903 G3 语音语种识别

本仓库用于 AIE1903 第三组（G3）的期中小组项目，任务是从一段语音录音中
识别说话者使用的语言。项目基线采用 Meta 的 MMS 语种识别模型
`facebook/mms-lid-126`。

> 当前状态（2026-10-27）：`scripts/` 里是推理和评测，没有训练脚本。
> 网页走 `pipeline.py`。`predict.py` 不走这五步，只做官方整段识别。
> 原始 wav 和模型权重不入库。PPT 不在本仓库。代码贡献见下面的 Contribution。
>
> 怎么跑：见 `docs/CODE.md` 与 `scripts/README.md`。

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
├── notebooks/   # 占位，里面没有实验
├── outputs/     # 生成结果（不提交到 Git）
├── scripts/     # 推理、网页和评测。没有训练
└── src/         # 占位，里面没有可运行代码
```

## 环境配置

我们实际跑通的环境是 Python 3.10、`torch 2.14.0+cpu`、`transformers 5.20.0`。
`requirements.txt` 允许安装这个版本。`denoiser` 不在这份清单里。
只有网页走到「又吵且整段分数低于 0.5」时才会 `import denoiser`，平时不需要。

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

- [x] 确定数据集、语言范围以及训练/验证/测试划分。（AISHELL `cmn`、LibriSpeech `eng`、MINDS-14 `kor`；不做训练划分，只用官方 checkpoint 推理）
- [x] 复现 `facebook/mms-lid-126` 基线预测。（`scripts/run_lid.py`、`scripts/predict.py`）
- [x] 确定主要评估指标，并统计各语言的分类结果。（Top-1 / 分数；挑战集 HIT/REJECT/MISS，见 `docs/results/`）
- [x] 完成一项扩展实验或控制变量对照实验。（`pipeline.py` 的五步检查；更早的 `eval_improved.py` 是切窗实验，网页不使用它）
- [x] 补充错误分析并记录方法局限性。（0.5 秒中文标 `uncertain`，内部最高分是 `mya`；1 秒中文接 1 秒英文可标 `mixed`；现场麦普通话被标成 `yue`，分数高，不是 `uncertain`）
- [ ] 完成最终报告和课堂展示。（PPT 不在本仓库）

## 复杂度

- 冻结模型：Wav2Vec2 序列分类，大约 10 亿参数。1280 维投影到 1024，再接到 126 个语言。
- 我们的代码：`scripts/` 下 8 个 Python 文件，大约 1200 行，没有训练循环。
- 网页最多用 8 个 CPU 线程。`predict.py`、`run_lid.py`、`eval_challenges.py`、`eval_improved.py`、`run_third_lang.py` 仍固定 1 个线程。
- 一条大约 4 秒的录音，在 CPU 上常常要十几秒。没有微调这 10 亿参数。

## Contribution

代码由 Fan Rui 完成，占本组工作的绝大部分。另外三人负责汇报，不负责本仓库里的源代码。PPT 文件不在本仓库。

| 成员 | 代码 | 汇报 |
|------|------|------|
| Fan Rui | `scripts/` 下全部推理、网页和评测，以及本 README、`docs/CODE.md`、`scripts/README.md` | 论文难点、预处理检查、下半学期只训练小头 |
| Han Sihao | 无源代码 | 论文和模型：MMS、mean-pool、为什么是 126 |
| Qian Chen | 无源代码 | 任务：输入、输出，以及为什么要先判断语言 |
| Zhu Zhiming | 无源代码 | 实现、命令行和网页演示 |
