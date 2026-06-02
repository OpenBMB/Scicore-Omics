<div align="center">

# SciCore-Omics

**面向空间组学与病理推理的基因感知多模态基础模型**

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-openbmb%2FSciCore--Omics-yellow)](https://huggingface.co/openbmb/SciCore-Omics)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Demo-SciCore--Omics-blue)](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/OpenBMB/Scicore-Omics?style=social)](https://github.com/OpenBMB/Scicore-Omics)

[模型权重](https://huggingface.co/openbmb/SciCore-Omics) ·
[在线 Demo](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics) ·
[快速开始](#快速开始) ·
[训练流程](#训练流程) ·
[引用](#引用)

[English](README.md) | 中文

</div>

---

## 最新动态

* **2026-06**：SciCore-Omics 模型权重已在 Hugging Face 发布：[openbmb/SciCore-Omics](https://huggingface.co/openbmb/SciCore-Omics)。
* **2026-06**：在线 Demo 已在 Hugging Face Space 开放：[Alkaidxxy/SciCore-Omics](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics)。
* **2026-06**：模型代码、训练入口与推理示例已公开。

---

## 简介

**SciCore-Omics** 是一个面向空间组学和病理推理的基因感知多模态基础模型，能够联合处理 **组织学图像**、**自然语言** 和 **转录组表达谱**。

该模型构建在 MiniCPM-V 多模态模型栈之上，并引入了专门的基因分支：首先使用 **NicheFormer** 对基因表达谱进行编码，然后通过 **Gene Q-Former** 压缩基因表示，最后利用 **Gene Projector** 将基因嵌入映射到语言模型的 token 空间中。

SciCore-Omics 的目标是让模型能够在统一的自回归语言模型框架下，同时理解组织形态、分子信号和生物学语义，从而支持空间组学和病理场景中的联合推理。

<p align="center">
  <img src="figs/fig1.png" width="90%">
</p>

---

## 核心贡献

* **基因感知的三模态基础模型**
  SciCore-Omics 将 MiniCPM-V 从图像-文本建模扩展到基因-图像-文本联合推理，并显式引入转录组输入通路。

* **专门的基因表示桥接模块**
  模型使用 NicheFormer、Gene Q-Former 和 Gene Projector，将可变长度的基因表达信号转换为固定长度、可被大语言模型接收的嵌入表示。

* **分阶段训练流程**
  仓库提供了基因桥接蒸馏、基于 Swift 的 CPT/SFT，以及 GSPO/PPO 风格的强化学习优化流程。

* **面向空间组学和病理场景**
  模型适用于需要同时解释组织形态与分子状态的任务，例如空间域识别、基因表达相关推理、组织状态描述和病理相关生物学解释。

* **公开发布路径完整**
  本仓库提供模型代码、公开权重、在线 Demo、环境配置和训练入口，便于研究者复现与扩展。

---

## 模型发布

| 项目      | 状态  | 链接                                                                               |
| ------- | --- | -------------------------------------------------------------------------------- |
| 模型权重    | 已发布 | [openbmb/SciCore-Omics](https://huggingface.co/openbmb/SciCore-Omics)            |
| 在线 Demo | 已发布 | [Alkaidxxy/SciCore-Omics](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics) |
| 源代码     | 已发布 | [OpenBMB/Scicore-Omics](https://github.com/OpenBMB/Scicore-Omics)                |
| 训练代码    | 已发布 | `train-distill-gene/`、`train-swift-cpt-sft/`、`train-rl/`                         |
| 许可证     | 已发布 | [Apache-2.0](LICENSE)                                                            |

---

## 核心思路

SciCore-Omics 在 MiniCPM-V 风格的视觉语言模型基础上增加了一条转录组输入通路：

```text
gene expression (.h5ad)
  -> gene tokenizer
  -> NicheFormer gene encoder
  -> Gene Q-Former bridge
  -> Gene Projector
  -> span embeddings in the LLM token space

image
  -> vision tower
  -> resampler
  -> span embeddings in the LLM token space

text prompt
  -> tokenizer

all modalities
  -> merged input embeddings
  -> MiniCPM-V / Qwen2 language model
  -> biological natural-language response
```

这种设计使模型既可以单独接收转录组上下文，也可以同时结合组织学图像和文本指令进行推理，同时保持标准的自回归语言模型接口。

需要强调的是，SciCore-Omics 并不是简单地把基因信号转换成普通文本，而是将基因表达谱编码为多模态嵌入，并插入到语言模型的 token embedding 序列中，从而使模型能够在统一表示空间内联合理解形态与分子信息。

---

## 仓库内容

本仓库主要包含以下代码模块：

| 路径                     | 作用                                                               |
| ---------------------- | ---------------------------------------------------------------- |
| `model/`               | 基因感知 MiniCPM-V 变体的核心模型、processor 和 tokenizer 定义                  |
| `eval/`                | 推理与评测示例                                                          |
| `train-distill-gene/`  | 基因桥接模块蒸馏代码，用于训练 `gene_qformer` 和 `gene_projector`，并支持将权重注入完整模型目录 |
| `train-swift-cpt-sft/` | 基于 `ms-swift` 的 CPT/SFT 示例脚本，以及面向基因感知 MiniCPM-V 工作流的自定义注册文件      |
| `train-rl/`            | GSPO/PPO 风格的分数引导强化学习优化流程                                         |
| `environment.yml`      | Conda 环境配置文件                                                     |
| `figs/`                | README 和文档中使用的图示                                                 |

如果你是第一次阅读本仓库，推荐顺序如下：

1. `eval/`：先跑通最小推理示例；
2. `model/`：理解模型结构；
3. `train-distill-gene/`：理解基因桥接模块的训练；
4. `train-swift-cpt-sft/`：理解 CPT/SFT 训练流程；
5. `train-rl/`：理解强化学习优化流程。

---

## 快速开始

### 1. 在线体验

最简单的方式是直接使用 Hugging Face Space：

```text
https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics
```

---

### 2. 安装环境

```bash
git clone https://github.com/OpenBMB/Scicore-Omics.git
cd Scicore-Omics

conda env create -f environment.yml
conda activate OMICS
```

参考环境为 Linux + NVIDIA A800-SXM4-80GB GPU。

需要注意的是，`flash-attn` 对 CUDA、PyTorch 和 GPU 环境较为敏感。如果安装失败，请根据本地 CUDA 和 PyTorch 版本调整 `flash-attn` 的安装方式。

---

### 3. 下载模型权重

可以使用 Hugging Face CLI 下载权重：

```bash
huggingface-cli download openbmb/SciCore-Omics \
  --local-dir ./weights/SciCore-Omics
```

也可以在代码中直接指定模型路径：

```text
openbmb/SciCore-Omics
```

---

### 4. 运行本地推理

示例命令如下：

```bash
python eval/example.py \
  --model_path ./weights/SciCore-Omics \
  --image_path examples/assets/example.png \
  --gene_path examples/assets/example.h5ad \
  --prompt "Please describe the tissue morphology and molecular state of this sample."
```

预期输出为一段自然语言回复，描述输入样本的组织形态、转录组上下文以及可能相关的生物学过程。

如果当前 `eval/example.py` 仍使用硬编码路径，请先将其中的模型路径、图像路径和 `.h5ad` 文件路径替换为你自己的本地路径。

---

## 输入格式

SciCore-Omics 支持三类输入：

| 输入类型  | 格式                    | 说明             |
| ----- | --------------------- | -------------- |
| 图像    | `.png`、`.jpg`、`.jpeg` | 组织学图像或组织 patch |
| 基因表达谱 | `.h5ad`               | 空间转录组表达矩阵      |
| 文本    | 自然语言                  | 用户问题或生物学指令     |

`.h5ad` 文件通常应包含：

```text
adata.X          # gene-expression matrix
adata.var_names  # gene names
adata.obs        # optional spot/cell metadata
adata.obsm       # optional spatial coordinates
```

基因名称需要与以下目录中的 tokenizer 资源兼容：

```text
model/gene_tokenizer/
```

---

## 模型架构

SciCore-Omics 的核心实现位于 `model/` 目录中。

### `model/` 中的关键组件

| 文件                                   | 作用                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `model/configuration_minicpm.py`     | 定义 `MiniCPMVConfig`，在 `Qwen2Config` 基础上扩展 `vision_config`、`slice_config` 和 `gene_config` |
| `model/configuration_nicheformer.py` | 定义基因编码器 NicheFormer 的配置类 `NicheformerConfig`                                             |
| `model/modeling_nicheformer.py`      | 实现基于 transformer 的 `NicheformerModel`，用于编码基因 token 序列                                    |
| `model/gene_qformer_module.py`       | 实现 `GeneQFormerBiomedBERT`，用于将可变长度基因 token 序列压缩为固定数量的 query token                        |
| `model/gene_projector_module.py`     | 将 Q-Former 输出从桥接模块隐藏维度映射到语言模型嵌入维度                                                        |
| `model/modeling_minicpmv.py`         | 整合 LLM、vision tower、resampler、NicheFormer、Gene Q-Former 和 Gene Projector                 |
| `model/processing_minicpmv.py`       | 实现 processor，将文本、图像和基因输入打包为模型可用的张量                                                       |
| `model/gene_tokenizer/`              | 基因 tokenizer 资源、词表和参考文件                                                                  |

---

## 基因分支连接方式

基因分支的整体流程如下：

1. 将基因表达谱 token 化为基因 token 序列；
2. 使用 `NicheformerModel` 将该序列编码为上下文化基因嵌入；
3. 使用 `GeneQFormerBiomedBERT` 将可变长度基因嵌入压缩为固定数量的 query token；
4. 使用 `GeneProjector` 将桥接输出映射到 MiniCPM-V 语言模型的 hidden space；
5. 将投影后的基因嵌入插入到语言模型输入流中，对应文本占位符 token span 的位置。

多模态融合发生在 MiniCPM-V 建模逻辑内部。图像特征和基因特征都会被转换为 embedding span，并在语言模型 forward 或 generation 前被 scatter 到最终的 `inputs_embeds` 序列中。

---

## 训练流程

SciCore-Omics 采用分阶段训练设计，而不是单一的端到端训练脚本。

整体流程包括：

1. **基因桥接蒸馏**：首先将转录组表示与语言模型空间对齐；
2. **基于 Swift 的 CPT/SFT**：将多模态模型适配到指令数据；
3. **强化学习优化**：通过分数引导 rollout 进一步优化部分模块的输出质量和格式稳定性。

---

### 1. 基因桥接蒸馏：`train-distill-gene/`

`train-distill-gene/` 目录用于单独训练基因桥接模块，包括：

* `gene_qformer`
* `gene_projector`
* 在更完整训练路径中可选的 `gene_cls_head`

这一阶段适用于核心多模态模型已经存在，但基因分支仍需要进一步对齐语言模型表示空间的场景。

主要脚本如下：

| 文件                                                               | 作用                               |
| ---------------------------------------------------------------- | -------------------------------- |
| `train-distill-gene/train_gene_bridge_distill.py`                | 最简单的单卡基因桥接蒸馏脚本                   |
| `train-distill-gene/train_gene_bridge_distill_ddp.py`            | 支持跨卡负样本的分布式蒸馏版本                  |
| `train-distill-gene/train_gene_bridge_distill_real_processor.py` | 当前推荐的训练路径，使用真实 processor 和参考基因对齐 |
| `train-distill-gene/inject_gene_bridge_weights.py`               | 将训练好的桥接模块权重注入完整 sharded model 目录 |

---

### 2. CPT/SFT 训练：`train-swift-cpt-sft/`

`train-swift-cpt-sft/` 目录包含基于 Swift 的 CPT/SFT 训练入口，用于训练基因感知 MiniCPM-V 模型。

这些脚本直接调用 `ms-swift` 框架：

```bash
swift pt
swift sft
```

基因相关逻辑通过 Swift 的自定义注册机制注入，而不是修改 Swift 框架源码。也就是说，训练脚本调用 `swift pt` 或 `swift sft`，并通过 `--custom_register_path` 传入自定义注册文件。

自定义注册文件位于：

```text
train-swift-cpt-sft/register/my_register_qformer.py
```

该文件定义了 `minicpm_v2_6_gene` 模型和模板路径，并包含基因输入处理逻辑。它会读取 `.h5ad` 基因输入、tokenize 基因名称、构建 `gene_input_ids`、`gene_attention_mask` 和 `gene_bound`，将基因占位符扩展为 Q-Former gene span，并将相关字段暴露给模型 batch。

主要文件如下：

| 文件 / 文件夹                                              | 作用                                                      |
| ----------------------------------------------------- | ------------------------------------------------------- |
| `train-swift-cpt-sft/register/my_register_qformer.py` | 面向基因感知 MiniCPM-V + Q-Former 模型的 Swift 自定义注册文件           |
| `train-swift-cpt-sft/script/cpt-example.sh`           | 使用自定义注册路径的 continued pretraining 示例脚本                   |
| `train-swift-cpt-sft/script/sft-example.sh`           | 使用 LoRA、gene/Q-Former target modules 和自定义注册路径的 SFT 示例脚本 |

运行前请替换脚本中的占位路径，例如：

```text
BASE_MODEL
DATA_DIR
OUTPUT_PATH
LOG_PATH
GENE_VOCAB_PATH
```

---

### 3. 强化学习优化：`train-rl/`

`train-rl/` 目录包含 GSPO/PPO 风格的强化学习流程，用于分数引导的多模态优化。

该流程将 rollout 生成、参考模型打分和分布式 actor 更新分离开来。

主要文件如下：

| 文件                          | 作用                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `train-rl/gen_worker.py`    | 采样样本、构建候选 batch、处理 rollout 数据，并计算 old-policy token log probabilities                              |
| `train-rl/ref_server.py`    | 启动 Flask reference server，恢复图像/基因/文本张量，并计算参考模型 token log probabilities                            |
| `train-rl/finetune_gspo.py` | 运行 DDP 训练循环，从 reference server 拉取 rollout batch，并使用带 KL 惩罚的 clipped GSPO/PPO-style objective 进行优化 |

RL 脚本默认冻结完整模型，并根据当前 rollout 是否包含基因或图像输入，选择性训练基因桥接模块、图像 resampler 和最后的 LLM 层。

---

## 推荐阅读入口

如果你的目标是：

* **理解模型架构**：从 `model/` 开始；
* **对齐或改进基因桥接模块**：从 `train-distill-gene/` 开始；
* **使用 Swift 自定义注册进行 CPT/SFT**：从 `train-swift-cpt-sft/` 开始；
* **进行分数引导的强化学习优化**：从 `train-rl/` 开始；
* **快速运行推理**：从 `eval/` 开始。

---

## 负责任使用与局限性

SciCore-Omics 面向空间生物学、病理 AI 和多模态生物医学建模研究发布。

该模型**不应作为独立临床诊断系统使用**。模型输出可能包含错误、不完整的生物学解释或未经验证的假设。任何生物医学或临床结论都应由具备资质的领域专家审阅，并通过适当的实验或临床证据进行验证。

潜在局限包括：

* 对图像质量和预处理方式敏感；
* 对基因名称、基因词表和 `.h5ad` 格式敏感；
* 可能生成不准确或过度推断的生物学解释；
* 对训练数据未充分覆盖的组织、疾病或测序平台泛化能力有限；
* 尚不具备前瞻性临床验证。

---

## 引用

如果你认为 SciCore-Omics 对你的研究有帮助，欢迎引用我们的工作：

```bibtex
@misc{xiao2026scicoreomics,
  title        = {SciCore-Omics: a tri-modal foundation model unifying histology, spatial transcriptomics and language for spatial biology},
  author       = {Xiao, Xinyu and Li, Yunfei and Zeng, Zheni and others},
  year         = {2026},
  note         = {Manuscript in preparation}
}
```

正式引用信息将在论文公开后更新。

---

## 联系方式

如果你有问题、建议或 bug 反馈，欢迎在 GitHub 仓库中提交 issue，或通过邮件联系：

* Xinyu Xiao: [xinyuxiao1@outlook.com](mailto:xinyuxiao1@outlook.com)

---

## 许可证

本项目基于 [Apache-2.0 License](LICENSE) 开源发布。
