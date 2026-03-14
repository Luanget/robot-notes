---
title: latent token
---

# latent token

## 1. latent token 是什么

在 ACT 中，latent token 不是图像 token，也不是机器人状态 token。
它表示的是：

> 当前这条动作序列所对应的“动作模式条件”

更具体地说，latent token 是由一个潜变量 `z` 经过线性投影得到的 transformer token。
这个 token 会与 proprio token 和 image tokens 一起进入主 encoder。

因此，latent token 的作用不是直接输出动作，而是：

> 为主 transformer 提供一个额外的条件，用来区分不同的合理动作模式。

## 2. 为什么需要 latent token

ACT 处理的是动作序列预测问题。
在很多机器人任务中，同样的观测下，往往不只有一种正确动作。

例如：

- 抓取一个物体时，可以略微从左靠近
- 也可以略微从右靠近
- 两种轨迹都可能成功

如果模型只有“观测 → 动作”这一条确定性映射，那么它容易把这些不同模式平均起来，得到一个折中的动作结果。
这种“平均动作”反而可能是不好的。

因此，ACT 引入 latent token 的目的就是：

> 用一个低维潜变量来表示“这次要走哪一种动作模式”。

这样，主 transformer 在生成动作时，就不是只依赖图像和状态，而是依赖：

- 图像
- 当前机器人状态
- latent 所表示的动作模式条件

## 3. latent token 从哪里来

latent token 并不是直接从图像产生的。
它来自 CVAE encoder。

训练时，CVAE encoder 的输入主要包括：

- 当前状态 `qpos`
- 真实动作序列 `actions`

编码过程可以概括为：

1. 把 `actions` 投影为 action embeddings
2. 把 `qpos` 投影为一个状态 token
3. 在序列最前面加入一个 `CLS token`
4. 把整个序列送入 encoder
5. 取 `CLS token` 输出作为全局摘要
6. 经过线性层映射为 `mu` 和 `logvar`
7. 采样得到潜变量 `z`
8. 再把 `z` 投影成 latent token

因此，latent token 的真正来源是：

> `qpos + actions` 经过 latent encoder 后得到的潜变量表示

## 4. latent token 的生成过程

下面按步骤详细写一次。

### 4.1 构造输入序列

训练时会构造如下序列：

```text
[CLS, qpos, a_1, a_2, ..., a_T]
```

其中：

- `CLS`：全局摘要 token
- `qpos`：当前机器人状态 token
- `a_t`：第 `t` 个动作位置对应的 token

这里的动作序列并不是直接拿原始数值进 transformer，而是先投影到 hidden dimension。

### 4.2 encoder 聚合全局信息

这个序列经过 latent encoder 后，每个 token 都会与其他 token 发生 self-attention。
最终，`CLS token` 会变成对整条序列的全局摘要。

可以理解为：

> `CLS token` 读完了当前状态和整条动作序列，并把它们压缩成一个摘要表示

### 4.3 得到 `mu` 和 `logvar`

接下来，对 `CLS token` 输出做线性映射，得到：

- `mu`
- `logvar`

它们共同定义了一个高斯分布：

\[
q(z|x,a) = \mathcal{N}(\mu, \sigma^2)
\]

其中：

\[
\sigma = \exp(0.5 \cdot \text{logvar})
\]

### 4.4 重参数化采样得到 `z`

训练时并不是直接用 `mu` 作为 latent，而是通过重参数化技巧采样：

\[
z = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
\]

这样做的好处是：

- 允许随机采样
- 又能保持反向传播可行

### 4.5 把 `z` 投影成 latent token

采样得到的 `z` 还是一个潜变量向量，并不是 transformer token。
因此还要经过一个投影层，把它映射到 hidden dimension：

\[
\text{latent\_input} = W z + b
\]

这个 `latent_input` 才是后面进入主 transformer 的 latent token。

## 5. latent token 在主 transformer 中怎么用

得到 latent token 后，它不会单独拿去预测动作，而是作为 encoder 输入序列中的一个特殊 token。

主 encoder 输入通常是：

```text
[latent, proprio, img_1, img_2, ..., img_HW]
```

也就是说，latent token 与：

- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- [[RoboTwin/ACT/concepts/image-tokens|image tokens]]

一起进入主 encoder。

其作用是：

> 让后续所有 token 在 self-attention 中都能访问到“当前动作模式条件”

这样，图像 token 和状态 token 不再只是根据环境和姿态编码，而是能够在编码阶段就知道：

> 当前动作应该偏向哪一种行为模式

## 6. latent token 在训练时和推理时的区别

这是 ACT 中非常关键的一点。

### 6.1 训练时
训练时有真实未来动作 `actions`，所以可以通过：

- `qpos`
- `actions`

构造 posterior，再采样得到 `z`，最后得到 latent token。

因此训练时的 latent token 是：

> 从真实动作序列中推断出来的 posterior 条件

### 6.2 推理时
推理时没有真实未来动作，因此不能再通过 posterior encoder 得到 `z`。

这时通常直接令：

\[
z = 0
\]

也就是使用标准高斯 prior 的中心点，再把它投影成 latent token。

因此推理时的 latent token 是：

> 默认 prior 点投影得到的条件 token

## 7. 为什么推理时 `z=0` 还能工作

表面上看，训练时 latent token 来源于真实动作，而推理时却直接用 `z=0`，似乎会不一致。
但这里有 KL 正则项在起作用。

训练时，模型不仅要最小化动作重建误差，还要最小化 posterior 与标准高斯之间的 KL 距离：

\[
D_{KL}(q(z|x,a)\,\|\,\mathcal{N}(0,I))
\]

这意味着：

- posterior 不能离标准高斯太远
- latent 空间会被约束到一个较规则的区域

因此，推理时取标准高斯中心点 `z=0`，通常仍然能对应到一个合理、稳定的动作模式。

所以更准确地说：

> 不是任意 z 都无所谓，而是训练时通过 KL 约束，使得推理时使用默认 prior 点也能工作。

## 8. latent token 在整体数据流中的位置

在 ACT 的整体前向中，latent token 的位置大致如下：

1. 输入 `qpos` 和 `actions`（训练时）
2. latent encoder 输出 `mu` 和 `logvar`
3. 采样得到 `z`
4. 投影得到 latent token
5. latent token 与 proprio token、image tokens 一起进入主 encoder
6. encoder 输出 memory
7. decoder 用 query 从 memory 中提取动作信息
8. 最终输出动作 chunk

因此，latent token 不是最终目标，而是整个动作生成过程中的条件桥梁。

详见 [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]。

## 9. latent token 的核心作用总结

我认为 latent token 的作用可以总结为三点：

### 9.1 表示动作模式
它不是环境信息，也不是当前姿态，而是“这条动作轨迹是什么风格/模式”。

### 9.2 避免平均动作
在多模态动作分布下，latent token 帮助模型区分不同合理动作，而不是把它们平均掉。

### 9.3 条件化主 transformer
它作为一个特殊 token 进入主 encoder，使后续动作预测从一开始就带有动作模式条件。

## 10. 容易混淆的点

### 10.1 latent token 不是 image token
它不是来自图像 backbone，而是来自 CVAE encoder。

### 10.2 latent token 不是直接的动作输出
它只是条件变量，不是最终动作。

### 10.3 latent token 和 `z` 不是同一个东西
- `z`：潜变量向量
- latent token：`z` 经过投影后，适合进入 transformer 的 token 表示

### 10.4 推理时 latent token 仍然存在
只是它不再来自 posterior，而是来自默认 prior 点。

## 11. 复习时最应该记住的点

1. latent token 来源于 `qpos + actions` 的 CVAE 编码结果
2. `CLS token` 输出经线性层得到 `mu` 和 `logvar`
3. 采样得到 `z` 后，还要再投影成 latent token
4. latent token 会进入主 encoder，而不是直接输出动作
5. 推理时没有真实动作，因此通常使用 `z=0`
6. KL 项保证推理时默认 prior 点仍然可用

## 12. 相关笔记跳转

- [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]
- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流转]]
- [[RoboTwin/ACT/detr_vae|DETRVAE]]
- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- [[RoboTwin/ACT/concepts/image-tokens|image tokens]]
- [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]
- [[RoboTwin/ACT/concepts/memory|memory]]
