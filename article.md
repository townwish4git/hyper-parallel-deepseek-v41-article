# HyperParallel 适配 DeepSeek-V4.1-Flash：拥抱前沿模型架构，释放昇腾大模型训练潜能

## 一、前言：DeepSeek-V4.1-Flash 发布，大模型训练迎来新挑战

随着大模型加速向长上下文、多模态与智能体应用演进，模型能力的提升不再仅仅依赖参数规模的增长，如何在有限的计算、存储和通信资源下实现更高效的训练与推理，正成为技术创新的重要方向。

2026 年 9 月，DeepSeek 正式发布 **DeepSeek-V4.1-Flash**。作为新一代多模态 MoE 模型，其拥有 **552B 骨干参数**，支持最长 **1M Tokens** 的上下文，并通过一系列架构与优化算法创新，实现了模型能力与计算效率的双重突破。

与上一代模型相比，DeepSeek-V4.1-Flash 引入了 Causal Encoder-Decoder（CED）非对称架构、Compressed Sparse Attention 2（CSA2）、Engram 条件记忆模块、Single-Pass mHC 等技术，并采用 Head-wise Muon 和 Sinkhorn Balancing 等优化方法，进一步探索大规模模型的高效训练路径。

这些创新提升了模型效率，也对训练框架提出了新的要求：如何支持跨层参数与计算状态共享？如何高效管理超大规模稀疏参数？如何将新型优化器与分布式训练结合？如何兼顾多模态、MoE 与长序列场景下的计算和通信效率？

**HyperParallel 是面向昇腾超节点的分布式并行加速框架**，以模型与系统优化解耦为设计理念，提供 FSDP、TP、CP、EP、PP 等并行能力，以及混合精度、重计算、异构内存管理与优化器等训练组件。

面向 DeepSeek-V4.1-Flash 的新架构与新需求，HyperParallel 开展模型训练适配，依托可组合的分布式并行能力和模块化系统设计，探索前沿模型在昇腾算力平台上的高效训练实践。

## 二、聚焦模型技术创新，HyperParallel 构建高效训练支撑

### 1. Causal Encoder-Decoder：非对称架构，提升计算效率

DeepSeek-V4.1-Flash 的重要创新之一，是采用 Causal Encoder-Decoder（CED）架构，将输入处理与输出生成划分为不同的计算路径。

传统 Decoder-only 模型在输入预填充与输出解码阶段通常使用相同的模型计算路径，而 CED 允许输入 Token 经过较轻量的 Encoder 路径处理，再通过 Decoder 完成输出生成。

得益于这一设计，DeepSeek-V4.1-Flash 在 Prefill 阶段每 Token 仅激活约 **8B 参数**，Decode 阶段激活约 **16B 参数**，显著降低了长输入场景的计算开销。

![DeepSeek-V4.1-Flash 总体架构：因果编码器、解码器及跨层 KV 路径](assets/article/deepseek-v41-architecture.png)


### 2. CSA2：跨层 KV 共享，挑战传统注意力计算模式

在长上下文场景下，Attention 的计算、状态保存与通信是影响模型效率的重要因素。

DeepSeek-V4.1-Flash 提出了 **Compressed Sparse Attention 2（CSA2）**，通过跨层共享 Global KV Cache 和稀疏索引，减少注意力状态的重复存储。

CSA2 包含三种工作模式：

- **Full**：计算并生成全局 KV，同时执行稀疏索引。
- **Reindex**：复用已有全局 KV，重新计算 Top-K 稀疏索引。
- **Reuse**：同时复用全局 KV 与 Top-K 索引，直接执行稀疏注意力计算。

CSA2 结合 FP4 KV Cache，使模型的全局 KV Cache 占用降低至约 **890 Bytes/Token**，约为 DeepSeek-V4-Flash 的四分之一。

![CSA2 的 Full、Reindex、Reuse 三种模式与共享状态](assets/article/csa2-modes.png)

![分层稀疏索引：首层建立候选池，后续 Reindex 层在池内选择 Top-512](assets/article/csa2-hierarchical-indexer.png)


### 3. Engram：超大规模条件记忆，探索高效参数管理

DeepSeek-V4.1-Flash 引入 **Engram 条件记忆模块**，通过基于 N-gram 的确定性哈希索引与稀疏查表机制，将部分知识记忆与 Transformer 主干计算解耦。

模型采用两个 Engram 模块，合计引入约 **196B 条件记忆参数**。

这一设计有助于提升模型能力，但超大规模 Embedding Table 也对训练系统的参数存储、数据访问及优化器状态管理提出了更高要求。

![Engram 官方架构图：N-gram 哈希查表、拼接与上下文门控](assets/article/engram-architecture.png)

### 4. Head-wise Muon 与 Sinkhorn：面向参数结构的新型优化器

DeepSeek-V4.1-Flash 在优化器设计方面也进行了针对性改进。

**Head-wise Muon**

传统 Muon 对整个权重矩阵执行更新变换，而 Head-wise Muon 将 Query、Key 等注意力权重按照 Attention Head 进行划分，再对不同 Head 分别应用 Muon 更新。

这种方式能够根据不同注意力头的梯度特征进行独立预条件化，更好地适应不同 Head 之间的异质性。

**Sinkhorn Balancing**

对于 Engram Embedding、Token Embedding 和 Prediction Head 等大规模矩阵，DeepSeek-V4.1-Flash 使用 Momentum + Sinkhorn Balancing 替代 Adam 类优化方式。

该方法通过迭代行、列归一化，平衡更新矩阵在不同维度上的尺度，并且只需维护动量缓冲区，有助于减少超大规模参数的优化器状态开销。

![Algorithm 1：结合 Sinkhorn Balancing 的动量更新算法伪代码](assets/article/sinkhorn-algorithm.png)


### 5. Single-Pass mHC：高效残差连接，简化计算路径

DeepSeek-V4.1-Flash 在 mHC（Manifold-Constrained Hyper-Connections）基础上引入 **Single-Pass mHC**。

该技术重新组织多路残差流的更新过程，将原本分散的残差更新、输入混合和系数预测整合到更高效的计算路径中。

配合 Mega-mHC 融合算子，模型推理阶段可以显著减少中间激活的数据读写开销。


### 6. 原生多模态 MoE：灵活组合并行策略，释放集群算力

DeepSeek-V4.1-Flash 从预训练阶段开始引入多模态数据，使用 DeepSeek-ViT 视觉编码器处理图像输入，并通过视觉投影模块与语言主干进行融合。

同时，模型延续细粒度 MoE 架构，并针对图像和文本 Token 分别维护专家负载均衡策略。

这种多模态与 MoE 相结合的结构，使得视觉计算、语言计算、专家通信以及长序列 Attention 需要在统一训练流程中协同执行。

可结合**模型图 1**底部的视觉与文本入口阅读本节：视觉特征经投影后与文本 Embedding 一起进入语言主干，各层 MoE 再承担条件计算。输入路径详见 [V4.1 报告 §2.1.1](https://arxiv.org/html/2609.19969v1#S2.SS1.SSS1)。


**模型不断创新，并行框架也需要随之演进。** HyperParallel 希望通过模型实现与系统优化解耦，让新架构能够复用成熟的并行基础能力，降低前沿模型的适配和训练成本。

## 三、训练实践：一键拉起 DeepSeek-V4.1-Flash 分布式训练

在完成模型架构与核心算子的适配后，HyperParallel 进一步提供了 DeepSeek-V4.1-Flash 的端到端训练示例，覆盖**纯文本训练、多模态训练以及多种分布式并行策略**。

依托统一的 Trainer 训练接口、声明式 YAML 配置和模块化并行策略，用户无需手动编排复杂的模型并行逻辑，即可通过启动脚本快速完成模型构建、数据准备、分布式初始化和训练执行。

下面以昇腾 NPU 环境为例，介绍 DeepSeek-V4.1-Flash 在 HyperParallel 上的训练流程。

### 1. 模型与环境准备

为便于快速验证新架构的训练能力，HyperParallel 提供了 DeepSeek-V4.1-Flash 的模型裁剪构建器 `build_cropped_deepseek_v41()`。

该构建器读取官方模型配置，在保留关键网络结构和层角色的基础上，对部分参数维度进行缩放，并通过 `HyperAutoModelForCausalLM.from_config()` 初始化模型。

当前训练示例支持以下两种场景：

| 配置项            | 纯文本训练    | 多模态训练    |
| -------------- | -------- | -------- |
| 语言模型层数         | 40 层     | 40 层     |
| 视觉模型层数         | —        | 32 层     |
| 参数维度缩放         | 1/8      | 1/4      |
| Routed Experts | 16       | 16       |
| 最大序列长度         | 4096     | 4096     |
| 默认训练步数         | 1        | 10       |
| 默认训练设备         | 16 卡 NPU | 16 卡 NPU |

裁剪过程保留模型的主干深度、指定层的功能以及核心计算结构，能够在更可控的资源规模下验证 CED、CSA2、Engram、MoE 等模块的训练流程。

需要说明的是，参数维度缩放并不等于模型总参数量按相同比例缩减。本示例从官方配置构建并随机初始化模型，主要用于训练链路验证，完整规模训练仍需另行进行资源配置和精度验证。

**环境安装**

首先获取并安装 HyperParallel：

```bash
git clone https://github.com/mindspore-ai/hyper-parallel.git
cd hyper-parallel

source /path/to/Ascend/cann/set_env.sh
python -m pip install -e .
```

运行前需准备与环境匹配的 PyTorch、torch-npu、CANN、Transformers，以及增强稀疏 Attention 所依赖的 `omni_training_custom_ops` 算子包。

准备官方模型配置与 Tokenizer，并设置模型目录：

```bash
export DSV41_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash
export PYTHON_BIN=python
```

完成上述准备后，即可开始训练。

### 2. 纯文本训练：支持 TP、CP 等多种并行配置

HyperParallel 提供了开箱即用的纯文本训练示例，能够自动生成本地测试数据和 Engram 元数据，完成模型初始化与分布式训练。

**Step 1：启动 TP1 基础训练**

以单机 16 卡为例，在 HyperParallel 根目录执行：

```bash
NPROC_PER_NODE=16 bash \
  examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  "$DSV41_MODEL_DIR" tp1 \
  --training.train_iters=1 \
  --checkpoint.restore_from=null
```

训练脚本自动完成以下流程：

1. **模型资产准备**：加载官方配置及 Tokenizer，按需生成裁剪后的 Engram 元数据。
2. **数据准备**：自动生成长文本 JSONL 数据，并在训练过程中在线分词。
3. **分布式初始化**：利用 `torch.distributed.run` 启动多个 NPU 训练进程，初始化分布式通信环境。
4. **模型构建与并行适配**：根据 YAML 配置完成模型初始化、组件替换、参数分片和并行策略部署。
5. **训练执行**：通过 `TextTrainer` 完成前向计算、反向传播、梯度同步和优化器更新。

训练结束后，输出 `run_tp1.log` 日志，并在脚本成功执行后生成 `tp1.success` 标记，便于后续训练验证。

**Step 2：切换 TP2 / CP2 并行策略**

基于统一的训练脚本，用户只需调整启动模式，即可切换不同的并行配置：

```bash
# TP2：张量并行
NPROC_PER_NODE=16 bash \
  examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  "$DSV41_MODEL_DIR" tp2 \
  --checkpoint.restore_from=null

# CP2：上下文并行
NPROC_PER_NODE=16 bash \
  examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
  "$DSV41_MODEL_DIR" cp2 \
  --checkpoint.restore_from=null
```

当前示例提供以下并行组合：

| 模式  | TP | CP | EP | FSDP 分片数 |
| --- | -: | -: | -: | -------: |
| TP1 |  1 |  1 | 16 |       16 |
| TP2 |  2 |  1 | 16 |        8 |
| CP2 |  1 |  2 | 16 |       16 |

其中，TP2 模式需要先通过 TP1 基础训练验证；CP2 模式则采用共享注意力 KV AllGather 路径。

通过不同并行模式的组合，HyperParallel 可对 DeepSeek-V4.1-Flash 的 Attention、MoE 及其他模块进行分布式训练验证，为后续更大规模的集群训练提供基础。

此外，用户还可根据实际设备数量调整 EP Size、FSDP 分片数和 Global Batch Size，实现训练资源的灵活配置。

### 3. 多模态训练：统一支持视觉与语言联合训练

除纯文本训练外，HyperParallel 同样提供 DeepSeek-V4.1-Flash 的图文多模态训练示例。

多模态训练基于 `VLMTrainer`，采用统一的训练框架管理视觉编码器、语言模型、数据预处理及参数优化过程。

**Step 1：准备图文数据**

当前示例使用 OpenAI Messages 格式的 JSONL 数据，支持文本与图片组合输入，例如：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述图片中的物体。"},
        {
          "type": "image_url",
          "image_url": {"url": "images/example.png"}
        }
      ]
    },
    {
      "role": "assistant",
      "content": "图片中是一只猫。"
    }
  ]
}
```

将各条 JSON 记录分别保存为 JSONL 文件，并准备对应图片，即可用于多模态训练。

训练过程中，框架自动完成图像预处理、文本 Tokenization、图文输入组织及训练标签构建。

**Step 2：启动多模态训练**

通过以下命令，可启动单机 16 卡、10 Step 的多模态训练：

```bash
RUN_NAME=guide_vlm_10steps NPROC_PER_NODE=16 \
  bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
  "$DSV41_MODEL_DIR" /path/to/v41-data/train.jsonl \
  --training.train_iters=10 \
  --checkpoint.restore_from=null
```

当前多模态配置采用 TP=1、CP=1、EP=8、FSDP 分片数为 8 的并行组合，结合 Muon + AdamW 优化器与 FP32 主参数机制，实现图文联合训练的完整执行流程。

HyperParallel 根据 YAML 中声明的 `plan_overrides`，自动完成 mHC、Engram、Attention 等组件替换，并绑定 CSA2 并行 Wrapper、Engram EP 查表和 MoE 专家并行计算。

从模型构建、分布式策略部署，到数据处理、梯度更新及 Checkpoint 管理，均可通过统一的 Trainer 流程执行，无需在训练入口中手动编排各模块的并行操作。

### 4. 训练结果：训练指标可视化与模型正确性验证

基于 **单机8卡 Ascend 910B3 NPU（单卡64 GiB HBM）**，HyperParallel 完成了 DeepSeek-V4.1-Flash 的纯文本及多模态训练验证。实验采用随机初始化的裁剪模型，保留关键模块与网络深度，使用合成数据验证端到端训练流程。

**（1）纯文本训练：多种并行策略验证**

TP1、TP2 + Sequence Parallel 和 CP2 三种并行配置均成功完成6个训练Step，Loss及梯度范数均保持有限值。

| 指标 | TP1 | TP2 + SP | CP2 |
|---|---:|---:|---:|
| 平均步时（s） | 19.57 | 38.71 | 23.30 |
| 吞吐（Tokens/s） | 1,674.77 | 846.56 | 1,406.62 |
| 峰值分配显存（GiB） | 39.08 | 24.42 | 20.65 |

本次实验中，TP1取得最高训练吞吐；CP2相比TP1降低约 **47%** 的峰值分配显存，平均步时增加约19%，体现了不同并行策略在训练性能与内存开销之间的灵活权衡。

![不同并行策略性能对比](assets/article/deepseek-v41-parallel-performance.png)

**（2）多模态训练：端到端训练链路验证**

多模态训练成功完成10次优化器更新，Loss由 **12.1633下降至10.5255**，训练过程中Loss及梯度范数均保持有限值。

| 指标 | 实测结果 |
|---|---:|
| 完成步数 | 10/10 |
| 平均步时 | 5.76 s |
| 峰值分配显存 | 9.25 GiB |
| 峰值保留显存 | 11.03 GiB |

![多模态训练Loss曲线](assets/article/deepseek-v41-vlm-loss.png)

![多模态训练日志](assets/article/deepseek-v41-training-log.png)

此外，本次实验成功保存了多模态训练Checkpoint，相关模型单元测试通过，进一步验证了模型组件与训练流程的可执行性。

**本次实测验证了HyperParallel对DeepSeek-V4.1-Flash纯文本、多模态及多维并行训练的端到端支持能力。** 实验结果基于裁剪模型、合成数据与短程训练，完整规模模型性能、长程收敛及Checkpoint恢复仍有待进一步验证。

*注：平均步时排除首步开销；纯文本吞吐按有效Token统计。多模态样本未填满4K上下文，两类训练的步时不宜直接比较。*


## 四、结语：让前沿模型创新更快落地

从 Causal Encoder-Decoder 到 CSA2，从 Engram 条件记忆到 Head-wise Muon、Sinkhorn Balancing，DeepSeek-V4.1-Flash 展示了大模型架构与训练优化算法的持续创新。

与此同时，模型结构日益复杂，也意味着训练框架需要具备更强的灵活性、可组合性和扩展能力。

**HyperParallel 持续围绕昇腾超节点算力架构，构建统一、高效、可扩展的分布式训练基础设施。** 通过模型与系统优化解耦、模块化适配及多维并行技术组合，为 MoE、多模态、超长序列和新型优化器等前沿技术提供训练支撑。

面向未来，HyperParallel 将继续完善复杂模型结构适配、分布式优化器、稀疏参数管理、混合精度以及通信计算重叠等能力，持续提升大规模训练的易用性与效率。

**让先进模型更易训练，让异构算力更高效释放。**

欢迎关注 HyperParallel 开源社区，共同探索大模型训练技术的更多可能！

**项目开源地址：**

https://github.com/mindspore-ai/hyper-parallel

**DeepSeek-V4.1-Flash 技术报告：**

https://arxiv.org/abs/2609.19969

**欢迎 Star、Fork，并通过 Issue 和 Pull Request 参与社区共建！**
