---
title: ACT
---

# ACT

这部分笔记围绕 RoboTwin 仓库中的 ACT 策略实现展开，目标不是只解释抽象概念，而是把：

- 整体数据流
- 训练 / 推理差异
- 模型模块关系
- 关键概念页
- 源码文件职责

组织成一套**适合复习、也适合回查源码**的结构化笔记。

如果你已经大致知道 transformer、CVAE、decoder query 这些术语，但一看到源码还是会混乱，那么建议按下面顺序阅读。

---

## 建议阅读顺序

### 1. 先看整体主线

- [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]
  - 这一篇解决的问题是：ACT 从输入到输出到底经过哪些模块，latent / backbone / transformer / action head / loss 在整体链路里分别处于什么位置。

- [[RoboTwin/ACT/train-and-infer|训练与推理流程]]
  - 这一篇解决的问题是：训练和推理到底哪里不同，为什么训练时有 posterior、推理时直接用 `z=0`，以及两条路径哪些共享、哪些分叉。

---

### 2. 再看模型主体

- [[RoboTwin/ACT/detr_vae|DETR-VAE 主体结构]]
  - 这一篇负责把 latent encoder、视觉 backbone、主 transformer、action head 串起来，并明确 CVAE encoder 和主 transformer encoder 不是同一个东西。

- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流]]
  - 这一篇重点解释主 transformer 内部的数据是怎么流动的：image tokens、proprio token、latent token 如何进入 encoder，query embeddings 如何在 decoder 中变成动作表示。

- [[RoboTwin/ACT/act_policy|ACT Policy 与损失组织]]
  - 这一篇负责解释外层 policy 如何调用模型、如何裁切动作 chunk、如何计算 L1 和 KL，以及训练阶段最终 loss 的组织方式。

---

### 3. 再看关键概念页

- [[RoboTwin/ACT/concepts/index|ACT 概念索引]]
  - 概念页适合在你已经看过主线后，用来做快速回忆和局部复习。

重点建议先看：

- [[RoboTwin/ACT/concepts/latent-token|latent token]]
- [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]
- [[RoboTwin/ACT/concepts/memory|memory]]
- [[RoboTwin/ACT/concepts/image-tokens|image tokens]]
- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- [[RoboTwin/ACT/concepts/action-head|action head]]

---

### 4. 最后看源码定位图

- [[RoboTwin/ACT/code-file-map|ACT 代码文件地图]]
  - 当你已经理解整体逻辑，但还经常忘记“某个功能到底在哪个文件里”时，这一篇最有用。
  - 它会帮你快速定位：
    - train / infer 分叉看哪里
    - latent 看哪里
    - query / transformer 看哪里
    - backbone 看哪里
    - loss 和 temporal aggregation 看哪里

---

## 这套笔记建议怎么用

如果你是第一次系统梳理 ACT，建议这样读：

1. `act-overall-dataflow.md`
2. `train-and-infer.md`
3. `detr_vae.md`
4. `transformer-dataflow.md`
5. `act_policy.md`
6. `concepts/` 下的关键概念页
7. `code-file-map.md`

如果你已经学过一次，后面复习时可以直接：

- 先看 [[RoboTwin/ACT/act-overall-dataflow|整体数据流]] 找回主线
- 再看 [[RoboTwin/ACT/code-file-map|代码文件地图]] 快速定位源码
- 对某个点模糊时，再跳到对应概念页

---

## 当前这部分笔记的目标

这套 ACT 笔记的目标不是替代源码，而是尽量做到：

> 在不立即打开源码的情况下，也能快速回忆：
> 当前模块在整体链路中的位置、它接收什么、输出什么、为什么要这样设计。

如果后面继续扩展，可以优先把以下内容接着补强：

- imitation / rollout 评估流程
- temporal aggregation 的细化说明
- 数据集样本组织与 `qpos / actions / is_pad / image` 的来源
- RoboTwin 中 ACT 与具体任务配置的关系
