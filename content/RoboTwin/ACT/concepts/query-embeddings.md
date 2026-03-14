---
title: query embeddings
---

# query embeddings

## 1. 在这份 RoboTwin 实现里，它具体对应什么

在这份仓库里，query embeddings 直接对应：

```python
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

其中：

- 文件：`policy/ACT/detr/models/detr_vae.py`
- `num_queries` 在构建模型时直接设为 `args.chunk_size`
- 所以 **query 的个数 = 一次前向要预测的动作 chunk 长度**

这意味着：

> 在这份实现里，“一个 query 对应一个动作槽位”不是抽象说法，而是代码层面直接成立的事实。

也就是说：

- 第 1 个 query 对应 chunk 里的第 1 个动作位置
- 第 2 个 query 对应 chunk 里的第 2 个动作位置
- ...
- 第 K 个 query 对应 chunk 里的第 K 个动作位置

这里的 query 不是输入 token，也不是历史动作，而是 decoder 端一组**可学习的输出槽位**。

---

## 2. 为什么 ACT 需要这组 query

ACT 不是一步只预测一个动作，而是一次预测一个 action chunk。

假设：

- `chunk_size = K`

那么模型一次前向就会输出未来 `K` 个动作。

问题在于：decoder 必须区分这 `K` 个输出位分别是谁负责的。
如果没有 query embeddings，decoder 只拿到一组空的 decoder token，就不知道：

- 哪个槽位负责最近一步动作
- 哪个槽位负责更远一步动作
- 各个输出位彼此如何区分

所以 query embeddings 的本质是：

> 给 decoder 一组带身份的输出槽位，让模型知道“我要生成第几个动作位置”。

这和 DETR 里的 object queries 非常像，只不过 DETR 里 query 对应检测槽位，这里对应的是**动作槽位**。

---

## 3. 为什么说“一个 query 对应一个动作槽位”

### 3.1 不是自回归输入

很多序列模型的 decoder 会把前一步真实输出或已生成输出喂进去，再生成下一步。
但这份 ACT 不是这样。

它的 decoder 更像是：

1. encoder 先把输入观测编码成 [[RoboTwin/ACT/concepts/memory|memory]]
2. decoder 再拿 `K` 个 learned queries 去查询这段 memory
3. 每个 query 负责抽取一个未来动作位置对应的信息

所以 query 不是“前一步动作”，而是：

> “请给我第 i 个动作位置需要的表示。”

### 3.2 `num_queries == chunk_size`

这点是最关键的实现级事实。

在本仓库里：

```python
num_queries=args.chunk_size
```

所以 query 个数和 chunk 长度直接绑定。

这就决定了：

- decoder 输出有多少个位置
- action head 最终要回归多少步动作

因此“query 对应动作槽位”在这里不是松散理解，而是模型结构本身的定义。

---

## 4. 为什么 decoder 输入是零初始化 `tgt`

在 `policy/ACT/detr/models/transformer.py` 中，decoder 前面有一行：

```python
tgt = torch.zeros_like(query_embed)
```

也就是说，decoder 的初始内容输入是全 0。

这时最容易困惑的是：

- `tgt` 都是 0
- 那 decoder 怎么区分不同动作槽位？

答案是：

> 靠的不是 `tgt`，而是 `query_embed`。

也就是说，decoder 一开始虽然“没有内容”，但每个槽位都带着不同的 query 身份信息进入注意力计算。

可以把它理解成：

- `tgt`：这个槽位当前携带的内容，初始化为空
- `query_embed`：这个槽位是谁，它负责哪个输出位置

所以这份实现的核心不是“把动作 token 喂进 decoder”，而是：

> 先给 decoder 一组空槽位，再用 query embeddings 指定这些槽位的身份。

---

## 5. 这份实现里 decoder 实际收到了什么

在 transformer forward 中，decoder 调用是：

```python
hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
```

所以 decoder 端真正拿到的是：

- `tgt`：全 0 初始化槽位
- `memory`：encoder 输出序列
- `pos`：encoder memory 的位置编码
- `query_pos`：query embeddings

这说明：

1. decoder 的“内容来源”主要来自 memory
2. decoder 的“槽位身份来源”主要来自 query embeddings

---

## 6. self-attention 和 cross-attention 在这里各做什么

### 6.1 self-attention：协调各个动作槽位

decoder 的 self-attention 发生在 query 槽位之间。

也就是说，它让：

- 第 1 个动作位
- 第 2 个动作位
- ...
- 第 K 个动作位

彼此交换信息。

这是必要的，因为 action chunk 不是 `K` 个完全独立的动作，而是一个连续动作片段。
比如：

- 第 1 步抬手
- 第 2 步接近目标
- 第 3 步闭合夹爪

这些动作之间天然有关联。

所以 self-attention 的作用可以记成：

> 在输出空间内部，让不同动作槽位彼此协调。

### 6.2 cross-attention：从 memory 中读取条件信息

decoder 的 cross-attention 则让每个 query 去读 encoder 输出的 [[RoboTwin/ACT/concepts/memory|memory]]。

而这份实现里的 memory 不是单个 pooled 向量，而是：

> encoder 输出的整段 token 序列。

这段序列中包含了：

- [[RoboTwin/ACT/concepts/latent-token|latent token]]
- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- [[RoboTwin/ACT/concepts/image-tokens|image tokens]]

所以每个 query 都会从整段 memory 中加权读取自己最需要的信息。

因此 cross-attention 的核心作用是：

> 让第 i 个动作槽位，从融合后的条件序列里提取对自己最有用的信息。

---

## 7. query 是怎样一步步变成动作表示的

### 第一步：初始化空槽位

- `query_embed` 提供槽位身份
- `tgt = 0` 提供初始空内容

### 第二步：query 之间先协调

self-attention 让不同动作位置知道彼此关系，避免每个位置各自独立乱预测。

### 第三步：从 memory 读取条件

cross-attention 让每个动作槽位从 encoder 输出序列中读取：

- 当前观察到的视觉信息
- 当前机器人状态
- latent 条件所表示的动作模式

### 第四步：逐层细化 hidden state

经过多层 decoder 后，每个 query 对应一个动作 hidden state。

在这份实现里，随后直接做：

```python
a_hat = self.action_head(hs)
```

也就是说：

> decoder 输出的每个 query hidden state，几乎就是动作回归前的最后表示。

这也说明这份实现里的 [[RoboTwin/ACT/concepts/action-head|action head]] 很轻，真正的建模主要已经在 transformer 中完成了。

---

## 8. query embeddings 在整体数据流中的位置

把它放回总流程里看：

1. 训练时，`qpos + actions` 进入 latent encoder，得到 [[RoboTwin/ACT/concepts/latent-token|latent token]]
2. 图像经过 backbone 和 `input_proj` 得到 [[RoboTwin/ACT/concepts/image-tokens|image tokens]]
3. `qpos` 经过线性层得到 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
4. 这些 token 一起进入主 encoder
5. encoder 输出整段 [[RoboTwin/ACT/concepts/memory|memory]]
6. decoder 读入零初始化 `tgt` 和 learned query embeddings
7. 每个 query 通过 self-attention / cross-attention 变成一个动作槽位表示
8. 最后通过 [[RoboTwin/ACT/concepts/action-head|action head]] 输出整个动作 chunk

所以 query embeddings 的位置非常明确：

> 它属于主 transformer 的 decoder 端，用来定义输出动作槽位，而不属于输入端、视觉端或 latent encoder 端。

---

## 9. 复习时最应该立刻反应出来的 5 点

1. 这份实现里 query embeddings 对应 `self.query_embed = nn.Embedding(num_queries, hidden_dim)`
2. `num_queries == chunk_size`，所以一个 query 对应 chunk 中一个动作位置
3. decoder 输入的 `tgt` 是全 0，但 `query_embed` 不是 0，它负责区分槽位身份
4. decoder 先让 query 之间 self-attention，再让 query 对 encoder memory 做 cross-attention
5. decoder 输出的 `hs` 会直接送入 `action_head` 得到动作预测

---

## 10. 一句话总结

> 在 RoboTwin 这份 ACT 实现里，query embeddings 是 decoder 端的 learned action slots；
> 每个 query 对应 chunk 中一个动作位置，带着自己的槽位身份从整段 encoder memory 中读取信息，最后被直接映射成对应位置的动作。
