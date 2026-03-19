---
title: ACT 核心链路
---

# ACT 核心链路

这篇笔记把原来的“整体数据流”和“训练与推理流程”合并成一条主线，只保留最重要的信息：

- 输入是什么
- latent / transformer / action head 分别做什么
- 训练与推理差别在哪
- loss 如何约束模型

想补细节时，再跳到：

- [[RoboTwin/ACT/detr_vae|DETR-VAE 主体结构]]
- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流]]
- [[RoboTwin/ACT/act_policy|ACT Policy 与损失组织]]
- [[RoboTwin/ACT/concepts/index|ACT 概念索引]]

---

## 1. ACT 在这份仓库里要做什么

ACT 的任务可以概括成一句话：

> 给定当前时刻的观测，直接预测接下来一段动作 chunk。

这里的当前观测包括：

- 图像 `image`
- 机器人状态 `qpos`

训练时还会额外给：

- 未来动作 `actions`
- padding 标记 `is_pad`

因此，ACT 不是一步一步自回归地出动作，而是一次输出一段未来动作。

---

## 2. 整体结构可以拆成哪几块

在 RoboTwin 这份实现里，ACT 可以拆成 6 个核心模块：

1. **观测输入**
   - 多相机图像
   - 当前机器人状态 `qpos`

2. **latent 路径**
   - 训练时由 `qpos + actions` 经过 posterior encoder 得到 `z`
   - 推理时没有真实动作，因此直接令 `z = 0`
   - `z` 再投影成 [[RoboTwin/ACT/concepts/latent-token|latent token]]

3. **视觉路径**
   - 图像进入共享 backbone
   - 得到视觉特征，再投影成 [[RoboTwin/ACT/concepts/image-tokens|image tokens]]

4. **状态路径**
   - `qpos` 线性投影成 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]

5. **主 transformer**
   - encoder 融合 latent / proprio / image 三类条件
   - 输出整段 [[RoboTwin/ACT/concepts/memory|memory]]
   - decoder 用 [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]] 从 memory 中提取动作信息

6. **动作输出与损失**
   - decoder 输出 `hs`
   - `hs` 经过 [[RoboTwin/ACT/concepts/action-head|action head]] 输出动作 chunk
   - 训练时用 L1 + KL 优化

---

## 3. 输入是怎样进入模型的

### 3.1 图像 `image`

多路相机共用同一个视觉 backbone。每路图像先独立提特征，再在后面统一送入 transformer。

你如果想看图像从 HDF5、Dataset、DataLoader 一直到 backbone 的完整过程，直接看：

- [[RoboTwin/ACT/dataset-and-dataloader|Dataset 与 DataLoader]]

### 3.2 机器人状态 `qpos`

`qpos` 表示当前机器人本体状态。它不会和图像混在一起，而是单独投影成一个 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]，作为 encoder 条件输入。

### 3.3 动作 `actions`

只有训练时才有。它不是给 decoder 做 teacher forcing，而是送到 posterior encoder，用来推断 latent `z`。

---

## 4. latent 在 ACT 里做什么

ACT 不是纯 transformer，还叠加了一个 CVAE 风格的 latent 条件。

### 训练时

训练时有真实未来动作，所以可以构造 posterior：

```text
qpos + actions -> posterior encoder -> mu, logvar -> z
```

再把 `z` 投影成 [[RoboTwin/ACT/concepts/latent-token|latent token]]，送入主 transformer。

### 推理时

推理没有未来动作，因此 posterior 分支不再存在，直接令：

```text
z = 0
```

然后一样投影成 latent token。

### 核心理解

训练和推理最大的区别，主要就在这里：

> **latent 的来源不同，但主干 transformer 结构基本相同。**

---

## 5. transformer 在 ACT 里怎么用

这部分是 ACT 最重要的主干。

### 5.1 encoder 做什么

encoder 负责把三类条件融合起来：

- latent token
- proprio token
- image tokens

输出的是一整段条件表示序列，也就是 [[RoboTwin/ACT/concepts/memory|memory]]。

### 5.2 decoder 做什么

decoder 不接收真实动作作为输入，而是接收一组 learned queries：

- 每个 query 对应一个动作槽位
- query 从 memory 中读取与该槽位对应的信息
- 最终形成这一段动作 chunk 的隐藏表示

这就是为什么 ACT 一次可以并行预测整个动作 chunk，而不是一步一步生成。

如果你想看这一段细节，直接跳：

- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流]]
- [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]

---

## 6. 训练和推理到底哪里不同

### 相同的部分

训练和推理共享这些主干：

- 图像 backbone
- qpos 投影
- 主 transformer encoder / decoder
- action head

### 不同的部分

真正的差别主要只有一条：

- **训练时**：有 posterior，latent 来自 `qpos + actions`
- **推理时**：没有 posterior，直接用 `z=0`

所以不要把训练 / 推理想成两套网络，它们本质上是同一个主干，只是在 latent 来源上分叉。

---

## 7. 输出动作是怎么得到的

decoder 输出的每个 query hidden state，都会经过同一个 [[RoboTwin/ACT/concepts/action-head|action head]]，映射成一个动作向量。

于是：

- 第 1 个 query -> 第 1 个动作槽位
- 第 2 个 query -> 第 2 个动作槽位
- ...
- 第 K 个 query -> 第 K 个动作槽位

这里的 `K` 就是 chunk size，也对应代码里的 `num_queries`。

---

## 8. loss 如何约束整个网络

训练时，外层 [[RoboTwin/ACT/act_policy|ACT Policy]] 会做两件最重要的事：

### 8.1 动作 L1 loss

模型输出动作 chunk `a_hat`，和真实动作 `actions` 做逐元素 L1。

但因为动作序列会 pad，所以 loss 只在非 padding 位置上计算。

### 8.2 KL loss

latent 分支会输出 `mu` 和 `logvar`，然后加 KL 正则，约束 posterior 不要偏离太远。

### 最终损失

可以粗略记成：

```text
loss = L1 + kl_weight * KL
```

所以 ACT 的训练不是单纯动作回归，而是：

> **动作回归 + latent 分布约束**

---

## 9. 一条最适合复习的主线

你可以把整个 ACT 压成下面这条线：

```text
image + qpos
-> image tokens + proprio token
-> (训练时再加 latent token)
-> transformer encoder 融合条件
-> memory
-> decoder queries 提取动作槽位信息
-> action head 输出动作 chunk
-> L1 + KL 训练
```

如果你只想记住最关键的一句，那就是：

> ACT = **CVAE 风格 latent 条件** + **DETR 风格 query decoder**，用当前观测一次预测未来动作 chunk。

---

## 10. 建议怎么继续看

如果你想继续精读，顺序建议是：

1. [[RoboTwin/ACT/dataset-and-dataloader|Dataset 与 DataLoader]]
2. [[RoboTwin/ACT/detr_vae|DETR-VAE 主体结构]]
3. [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流]]
4. [[RoboTwin/ACT/act_policy|ACT Policy 与损失组织]]
5. [[RoboTwin/ACT/concepts/index|ACT 概念索引]]
