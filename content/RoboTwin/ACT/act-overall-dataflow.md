---
title: ACT 整体数据流
---

# ACT 整体数据流

## 1. 这篇笔记要解决的问题

这篇笔记的目标不是只看某一个文件，而是把 ACT 从输入到输出的整体前向过程串起来，回答以下问题：

1. ACT 的输入到底有哪些
2. 训练时和推理时的数据流有什么不同
3. latent token 从哪里来
4. 图像和机器人状态如何进入 transformer
5. decoder 如何输出动作 chunk
6. loss 如何反过来约束整个网络

这篇笔记相当于 ACT 的总览图。
如果只看单个模块，很容易知道“某一段代码在干什么”，但不知道“它在整体链路里的位置”。
因此，这篇笔记的重点是把各模块之间的数据关系串起来。

## 2. ACT 的整体结构概览

ACT 可以粗略分成以下几部分：

1. 输入部分
   - 图像 `image`
   - 机器人状态 `qpos`
   - 动作序列 `actions`（仅训练时有）
   - padding 标记 `is_pad`（仅训练时有）

2. latent 路径
   - 训练时，`qpos + actions` 进入 CVAE encoder
   - 输出 `mu`、`logvar`
   - 采样得到 `z`
   - 经过投影得到 [[RoboTwin/ACT/concepts/latent-token|latent token]]

3. 视觉路径
   - 图像进入 backbone
   - 变成 [[RoboTwin/ACT/concepts/image-tokens|image tokens]]

4. 本体状态路径
   - `qpos` 经过线性层投影
   - 变成 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]

5. 主 transformer
   - encoder 输入由 latent token、proprio token、image tokens 共同组成
   - encoder 输出 [[RoboTwin/ACT/concepts/memory|memory]]
   - decoder 使用 [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]] 从 memory 中提取动作信息

6. 输出头
   - decoder 输出隐藏表示 `hs`
   - 经过 [[RoboTwin/ACT/concepts/action-head|action head]] 输出动作 chunk
   - 还可以经过 `is_pad_head` 输出 padding 预测

7. 损失
   - 动作重建误差 L1
   - KL 正则项

## 3. 输入组成

ACT 一次前向中，核心输入包括：

### 3.1 图像 `image`
来自一个或多个相机。
每路相机图像会经过 backbone 提取特征，形成视觉特征图，后续整理为 image tokens。

### 3.2 机器人状态 `qpos`
表示当前机器人本体状态，例如关节角、末端状态等。
它会经过线性映射，形成一个 proprio token。

### 3.3 动作序列 `actions`
仅在训练时存在。
训练阶段不仅有当前观测，还知道专家动作序列，因此可以用它来构造 latent posterior。
推理时没有真实未来动作，因此这一路不会走训练时那条 posterior 编码路径。

### 3.4 `is_pad`
用于标记动作序列哪些位置是 padding。
在训练中，计算动作损失时要忽略这些 padding 位置。

## 4. 训练时的数据流

训练时的数据流是 ACT 最完整的路径，因为它同时包含：

- latent posterior 的构造
- 主 transformer 的动作预测
- loss 的计算

### 4.1 第一步：构造 latent posterior

训练时，模型拿到：

- 当前状态 `qpos`
- 真实动作序列 `actions`

然后将它们送入 CVAE encoder。

更准确地说，输入序列会构造成：

- `CLS token`
- `qpos token`
- `action tokens`

经过 encoder 后，取 `CLS token` 对应的输出表示，映射得到：

- `mu`
- `logvar`

然后通过重参数化采样：

\[
z = \mu + \sigma \odot \epsilon
\]

再经过线性投影，得到 [[RoboTwin/ACT/concepts/latent-token|latent token]]。

这一步的作用是：

> 把当前这条专家动作序列的“动作模式”压缩成一个潜变量条件。

详见 [[RoboTwin/ACT/concepts/latent-token|latent token]] 和 [[RoboTwin/ACT/detr_vae|DETRVAE]]。

### 4.2 第二步：提取图像特征

图像进入视觉 backbone，输出特征图。
这些特征图经过 `input_proj` 投影到 transformer 的隐藏维度，形成统一维度的视觉表示。

如果有多路相机，那么多路相机特征会按实现方式拼接后一起送入主 transformer。

这一部分最终形成 [[RoboTwin/ACT/concepts/image-tokens|image tokens]]。

### 4.3 第三步：构造 proprio token

当前机器人状态 `qpos` 经过线性层投影后，得到一个固定维度向量。
这个向量被当成一个独立 token，称为 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]。

### 4.4 第四步：送入主 encoder

主 transformer 的 encoder 输入并不只是图像，而是一个联合 token 序列：

- [[RoboTwin/ACT/concepts/latent-token|latent token]]
- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- [[RoboTwin/ACT/concepts/image-tokens|image tokens]]

可以写成：

```text
[latent, proprio, img_1, img_2, ..., img_HW]
```

这里的核心思想是：

> 不把状态和 latent 在最后再拼接，而是一开始就作为 token 与视觉 token 一起进入 encoder。

这样做的好处是：

- 图像 token 可以从一开始就看到 proprio 和 latent 条件
- proprio token 可以感知全局视觉信息
- latent token 也可以与图像、状态交互

经过多层 self-attention 后，encoder 输出 [[RoboTwin/ACT/concepts/memory|memory]]。

### 4.5 第五步：decoder 用 query 提取动作信息

主 decoder 的输入不是“前一个动作”，而是：

- 全零初始化的 `tgt`
- [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]
- encoder 输出的 [[RoboTwin/ACT/concepts/memory|memory]]

这里每个 query 都对应未来动作序列中的一个位置，也就是一个“动作槽位”。

decoder 主要做三件事：

1. query 之间 self-attention
   让不同动作位置建立关系

2. query 对 memory 做 cross-attention
   让每个动作位置从 memory 中提取与自己最相关的信息

3. FFN
   把得到的隐藏表示进一步变换

最终 decoder 输出：

- 每个未来动作位置对应的隐藏表示 `hs`

### 4.6 第六步：输出动作 chunk

decoder 输出 `hs` 后，经过 [[RoboTwin/ACT/concepts/action-head|action head]] 投影到动作空间：

\[
\hat{a} = \text{action\_head}(hs)
\]

于是得到整个动作 chunk 的预测结果。

同时还可以通过 `is_pad_head` 预测每个位置是否为 padding。

### 4.7 第七步：计算损失

训练时通常有两部分损失：

#### 1）动作重建损失
预测动作与真实动作做 L1 损失：

\[
L_{action} = \| \hat{a} - a \|_1
\]

计算时要忽略 padding 位置。

#### 2）KL 损失
CVAE 的 posterior 由 `mu` 和 `logvar` 表示，需要用 KL 项把它约束到标准高斯附近：

\[
L_{KL} = D_{KL}(q(z|x,a)\,\|\,\mathcal{N}(0, I))
\]

#### 总损失
总损失通常写成：

\[
L = L_{action} + \lambda L_{KL}
\]

其中 `\lambda` 是 KL 权重。

## 5. 推理时的数据流

推理时和训练时最大的不同是：

> 没有真实未来动作 `actions`

因此，训练时那条“由 `actions + qpos` 构造 posterior”的路径不能再使用。

### 5.1 latent 的变化

推理时没有 `actions`，因此不能再通过 CVAE encoder 得到 posterior。
这时通常直接使用 prior 的中心点，也就是：

\[
z = 0
\]

再把它投影成 [[RoboTwin/ACT/concepts/latent-token|latent token]]。

这意味着：

- 训练时：latent token 来自 posterior 采样
- 推理时：latent token 来自固定默认 prior 点

之所以这样仍然可行，是因为训练时 KL 项会把 posterior 约束到标准高斯附近，使得推理时使用默认 prior 点仍然能得到可用动作模式。

### 5.2 其余路径基本不变

除了 latent 生成方式变化外，其余流程基本与训练时一致：

1. 图像进入 backbone，得到 image tokens
2. `qpos` 形成 proprio token
3. 主 encoder 融合这些 token 得到 memory
4. decoder 用 query 从 memory 中提取动作信息
5. action head 输出动作 chunk

因此可以说：

> ACT 训练和推理的主干结构基本相同，最关键差异在于 latent token 的来源不同。

## 6. ACT 的关键中间概念

### 6.1 [[RoboTwin/ACT/concepts/latent-token|latent token]]
由训练阶段的动作序列和当前状态编码而来，用于表示动作模式条件。
推理时则由默认 prior 点投影得到。

### 6.2 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
由 `qpos` 投影而来，表示当前机器人状态。

### 6.3 [[RoboTwin/ACT/concepts/image-tokens|image tokens]]
由图像 backbone 特征图整理而来，表示环境观察信息。

### 6.4 [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]
decoder 中的 learned queries，每个 query 对应未来动作序列中的一个位置。

### 6.5 [[RoboTwin/ACT/concepts/memory|memory]]
encoder 输出的一组上下文化表示，是 decoder 查询动作信息的基础。

### 6.6 [[RoboTwin/ACT/concepts/action-head|action head]]
把 decoder 输出的隐藏表示映射到动作空间，得到动作预测。

## 7. ACT 的核心理解

我认为 ACT 最核心的思想可以概括为三句话：

### 7.1 用 latent 表示动作模式
同样的观测条件下，往往存在多种合理动作轨迹。
latent 的作用是避免模型把这些不同模式平均掉，而是通过条件变量区分不同模式。

### 7.2 用 encoder 融合多模态条件
主 encoder 的输入不是单一路径，而是：

- latent 条件
- 机器人状态
- 图像特征

它通过 self-attention 把这些信息融合成统一 memory。

### 7.3 用 decoder 把动作位置作为 query
decoder 中每个 query 不代表图像区域，而代表未来动作序列中的一个位置。
因此，ACT 本质上是在做：

> “未来动作槽位”对“条件化 memory”的查询与解码

## 8. 复习时最应该记住的点

1. 训练时多了一条由 `actions + qpos` 构造 latent posterior 的路径
2. 推理时没有真实 `actions`，因此 latent 改为默认 prior 点
3. 主 encoder 的输入不只有图像，还有 latent token 和 proprio token
4. encoder 输出的是 memory，不是动作
5. decoder 的 query 对应未来动作位置，而不是图像位置
6. action head 才负责把隐藏表示映射到动作空间
7. loss 由动作重建项和 KL 项共同组成

## 9. 相关笔记跳转

- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流转]]
- [[RoboTwin/ACT/detr_vae|DETRVAE]]
- [ACT Policy](act_policy.md)
- [[RoboTwin/ACT/train-and-infer|训练与推理流程]]
- [[RoboTwin/ACT/concepts/latent-token|latent token]]
- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- [[RoboTwin/ACT/concepts/image-tokens|image tokens]]
- [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]
- [[RoboTwin/ACT/concepts/memory|memory]]
- [[RoboTwin/ACT/concepts/action-head|action head]]
