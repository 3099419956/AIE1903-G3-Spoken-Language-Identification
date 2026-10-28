# AIE1903 G3 语音语种识别

本仓库用于 AIE1903 第三组（G3）的期中小组项目，任务是从一段语音录音中
识别说话者使用的语言。项目基线采用 Meta 的 MMS 语种识别模型
`facebook/mms-lid-126`。

> 当前状态（2026-10-27）：`scripts/` 里是推理和评测，没有训练脚本。
> 网页走 `pipeline.py`。`predict.py` 不走这五步，只做官方整段识别。
> 原始 wav 和模型权重不入库。PPT 不在本仓库。代码贡献见下面的 Contribution。
>
> 怎么跑：见 `docs/CODE.md` 与 `scripts/README.md`。
> 展示内容和 15 分钟页面安排：见 `docs/MIDTERM.md`。

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

模型文件可能较大。默认使用 Python、Hugging Face 和 PyTorch 的标准缓存。
如果要复用已有的共享缓存，只需要设置一个目录；脚本不会再使用某个同学的
私人绝对路径：

```powershell
$env:AIE1903_CACHE_DIR = "D:\aie1903-cache"
```

不设置时，模型会下载到当前用户的标准缓存。第一次运行需要联网和足够磁盘空间。

只检查标准库部分（不需要下载模型）：

```powershell
python -m unittest discover -s tests -v
python -m compileall -q scripts
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

- 冻结模型：课程建议的 Wav2Vec2 语种分类 checkpoint，最后接到 126 个语言。
- 论文使用 MMS 1B 预训练模型并微调全部参数；本仓库只加载 126 语种 checkpoint 做推理，没有训练循环。
- 我们的代码：`scripts/` 下的推理、网页和评测模块，没有修改模型权重。
- 网页最多用 8 个 CPU 线程。`predict.py`、`run_lid.py`、`eval_challenges.py`、`eval_improved.py`、`run_third_lang.py` 仍固定 1 个线程。
- 一条大约 4 秒的录音，在 CPU 上常常要十几秒。我们没有训练或微调模型权重。

## Contribution

代码主体是 Fan Rui 的预处理、网页和整段评测。另外三人也各有脚本，并负责读论文和准备自己的汇报。PPT 文件不在本仓库。

| 成员 | 做了什么 |
|------|----------|
| Fan Rui | 代码：`pipeline.py`、`demo_server.py`、`demo/index.html`、`eval_pipeline.py`、`eval_improved.py`，以及本 README、`docs/CODE.md`、`scripts/README.md`。实验：短音频拒识、噪声门、dns48 和其他降噪的对比、现场麦克风录音。汇报：论文难点、预处理检查、下半学期只训练小头。 |
| Han Sihao | 代码：`run_third_lang.py`，在同一个冻结模型上检查韩语。文献：阅读 MMS 论文里 wav2vec 2.0 预训练、LID 头、mean-pool、126 种语言（FLEURS 与 VoxLingua-107 的并集）以及 Table 7 为何不在 CPU 上整模微调。汇报：论文和模型。 |
| Qian Chen | 代码：`eval_challenges.py`、`data/README.md`，整理短音频、噪声和中英拼接测试，并写明 AISHELL、LibriSpeech、MINDS-14 三条录音的来源。文献：对照论文第 6 节，说明许多 ASR 仍是单语，所以要先做语种识别，而不是转写。汇报：任务的输入、输出和意义。 |
| Zhu Zhiming | 代码：`predict.py`、`run_lid.py`，官方整段识别入口和中英基线。核对三段干净录音的 Top-1 大约是 0.999，并整理命令行和网页的运行步骤。汇报：实现、命令行和网页演示。 |
