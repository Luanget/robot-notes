---
title: query embeddings
---

# query embeddings

## 1. query embeddings 是什么

在 ACT 中，`query embeddings` 是 decoder 端的一组**可学习查询向量**。

它们不是来自输入图像，也不是来自机器人状态，更不是来自真实动作。  
它们是一组模型参数，在训练过程中不断更新，用来告诉 decoder：

> “我要从 encoder 输出的 memory 中，取出第 1 个动作槽位、第 2 个动作槽位、……、第 K 个动作槽位对应的信息。”

因此，query embeddings 的本质不是“动作本身”，而是：

- 一组固定数量的查询位
- 每个查询位对应动作 chunk 中的一个位置
- decoder 通过这些查询位，从 encoder memory 中提取与该位置动作有关的信息

---

## 2. 为什么 ACT 需要 query embeddings

ACT 不是一步只预测一个动作，而是一次预测一个 **action chunk**。  
假设 chunk size 是 `K`，那么模型一次前向就要输出：

- 第 1 个未来动作
- 第 2 个未来动作
- ...
- 第 K 个未来动作

这意味着模型需要在 decoder 端同时维护 `K` 个“输出槽位”。

如果没有 query embeddings，decoder 就不知道：

- 哪个输出位置负责第 1 步动作
- 哪个输出位置负责第 2 步动作
- 哪个输出位置负责第 3 步动作

也就是说，**模型需要一个显式的“槽位索引机制”**。  
在 DETR 风格结构里，这个机制就是 query embeddings。

所以可以把它粗略理解为：

- 第 `i` 个 query embedding
- 对应“我要预测 chunk 中第 `i` 个动作”

这就是为什么我说：

> query 对应的不是某个输入 token，而是某个输出动作槽位。

---

## 3. 一个 query 为什么可以对应一个动作槽位

这一点是理解 ACT decoder 的关键。

### 3.1 decoder 不是在“读入动作序列”

在很多序列模型里，decoder 的输入来自已经生成的前面 token。  
但 ACT 这里不是这样。

ACT 的 decoder 更像 DETR 的 object queries：

- encoder 先把输入观测编码成 memory
- decoder 再拿一组 learned queries 去“问” memory
- 每个 query 负责抽取一个输出位的信息

因此，query 的职责不是表示“当前已经生成了什么”，而是表示：

> “请给我第 i 个输出槽位需要的表示。”

---

### 3.2 动作 chunk 是有位置结构的

虽然 ACT 不是自回归逐步输出，但 chunk 内部依然有顺序：

- 第 1 个动作：离当前最近
- 第 2 个动作：更往后一点
- ...
- 第 K 个动作：更远的未来动作

因此，decoder 需要区分：

- “我要预测最近一步动作”
- “我要预测更远一步动作”

这组区分，不是靠输入真实动作完成的，而是靠不同 query embeddings 完成的。

所以训练后，每个 query 往往会学到一种**与时间槽位相关的功能偏置**：

- 有的 query 更偏向抽取“立刻要做什么”
- 有的 query 更偏向抽取“后续延续动作是什么”

注意，这不是说 query 天生带时间标签，而是训练中通过监督逐渐学出来的。

---

## 4. 为什么 decoder 输入是零初始化 `tgt`

在 PyTorch / DETR 风格实现里，decoder 常见写法类似：

```python
tgt = torch.zeros_like(query_embed)
hs = decoder(tgt, memory, query_pos=query_embed, ...)
```

这里最容易困惑的地方是：

- 既然 `tgt` 全是 0
- 那 decoder 靠什么区分不同输出槽位？

答案是：

> 靠 `query embeddings`，不是靠 `tgt` 本身。

---

### 4.1 `tgt` 和 `query embeddings` 不是一回事

在 decoder 中，通常有两部分东西：

- `tgt`：当前 decoder token 的内容表示
- `query_pos` / `query_embed`：给每个 decoder token 附加的位置/身份信息

在 ACT / DETR 风格里：

- `tgt` 初始可以全 0
- 但 `query embeddings` 必须是不同的、可学习的

这样每个 decoder 槽位虽然初始内容为空，但它们带着不同的“身份标签”进入注意力计算。

你可以把它理解为：

- `tgt` 表示“这个槽位目前携带了什么内容”
- `query embedding` 表示“这个槽位是谁，它负责什么位置”

初始化时内容为空，所以 `tgt = 0` 没问题；  
但身份不能丢，所以 query embeddings 不能没有。

---

### 4.2 为什么可以从零开始

因为 decoder 的目标不是复原输入，而是：

1. 带着 query 身份去读取 encoder memory
2. 从 memory 中抽取与该输出槽位对应的信息
3. 逐层更新自己的隐藏表示
4. 最终形成动作表示

所以初始时 decoder 槽位“没有内容”并不矛盾。  
它后续的内容，本来就是要通过 attention 从 memory 中逐步取出来的。

换句话说：

> `tgt=0` 表示“我先什么都不知道”，  
> `query embedding` 表示“但我知道我是第 i 个动作槽位”。

---

## 5. decoder 中 self-attention 和 cross-attention 各做什么

这是 query embeddings 真正发挥作用的地方。

ACT 的 decoder 可以粗略看成每层都在做两件事：

1. decoder 内部槽位之间相互交流
2. 每个槽位去 encoder memory 中读取观测信息

前者主要对应 self-attention，后者主要对应 cross-attention。

---

### 5.1 self-attention：让各个动作槽位彼此协调

decoder 的 self-attention 发生在 query 之间。

也就是说，参与注意力的是：

- 第 1 个动作槽位
- 第 2 个动作槽位
- ...
- 第 K 个动作槽位

它的作用不是去看图像，而是让输出槽位之间建立关系。

这很重要，因为 action chunk 不是独立的 K 个动作，而是一个连续动作片段。  
例如：

- 第 1 步抬手
- 第 2 步继续靠近
- 第 3 步闭合夹爪

这些动作之间显然不是互相独立的。  
所以 decoder 需要让不同 query 之间交换信息，形成更一致的整体计划。

因此，self-attention 的核心作用可以理解为：

> 在“输出空间”内部，协调各个动作槽位之间的关系。

---

### 5.2 cross-attention：让每个动作槽位去读 memory

decoder 的 cross-attention 发生在：

- query 侧：decoder 当前槽位表示
- key/value 侧：encoder 输出的 memory

而 memory 里已经融合了输入观测信息，例如：

- [latent token](latent-token.md)
- [proprio token](proprio-token.md)
- [image tokens](image-tokens.md)

也就是说，encoder 先把“当前观测 + 条件信息”处理成一个统一表征，  
decoder 再让每个 query 去其中读取自己需要的信息。

例如某个 query 可能更关注：

- 手和目标物体的位置关系
- 某个相机视角中的局部区域
- latent token 提供的高层动作模式
- proprio token 提供的当前机械臂状态

所以 cross-attention 的核心作用是：

> 对于第 i 个动作槽位，从 memory 中抽取最相关的信息，形成该槽位的动作表示。

---

## 6. query 是怎样一步步变成动作表示的

可以把这个过程拆成下面几步。

### 6.1 初始状态

decoder 一开始有两样东西：

- 零初始化的 `tgt`
- 一组 learned `query embeddings`

此时每个槽位还没有具体动作内容，但已经有不同身份。

---

### 6.2 经过 self-attention

不同 query 槽位先彼此交互。

这一步的结果是：

- 第 1 个槽位知道自己和第 2、第 3 个槽位的关系
- 第 2 个槽位也能感知整个 chunk 的内部结构
- 输出不再像彼此独立的 K 个头，而是更像一个有内部一致性的计划片段

---

### 6.3 经过 cross-attention

每个槽位拿着当前表示去读 encoder memory。

memory 中包含：

- 当前图像观测的视觉信息
- 当前机器人状态
- latent 条件信息

于是每个 query 会把自己更新成：

- “适合当前观测条件下，第 i 个动作位置应当输出什么”的隐藏表示

这时候 query 已经不只是一个抽象槽位了，  
而是逐渐变成了**面向具体任务场景的动作位表示**。

---

### 6.4 经过多层 decoder 反复细化

decoder 通常不只一层。

所以一个 query 的表示不是一次 cross-attention 就结束，而是会反复经历：

- 槽位间协调
- 从 memory 读取
- 更新自身表示

层数越往后，它越接近最终可用于动作回归的表示。

最终得到的 `hs` 可以理解为：

> 每个 query 对应一个已经融合了全局观测、状态条件、latent 条件和 chunk 内部依赖关系的动作隐藏向量。

---

### 6.5 送入 action head

decoder 输出的每个 query hidden state，接下来会经过 [action head](action-head.md)：

```text
query hidden state  ->  linear / MLP  ->  action_t
```

于是：

- 第 1 个 query 输出第 1 个动作
- 第 2 个 query 输出第 2 个动作
- ...
- 第 K 个 query 输出第 K 个动作

这就是“query 对应动作槽位”的最终落点。

---

## 7. query embeddings 和位置编码是什么关系

这个地方容易混淆。

在 encoder 里，我们常说图像 token 有位置编码，因为图像 patch 本身有空间位置。  
但 decoder 里的 query embeddings 更像是：

- 输出槽位的身份编码
- learned output slots
- learned positional anchors in output space

它们和图像位置编码不一样。

### encoder 位置编码强调的是：

- 这个 token 来自图像哪里
- 这个 token 在输入序列里位于什么位置

### decoder query embeddings 强调的是：

- 这个输出槽位是第几个
- 它要负责哪个动作位置

所以 query embeddings 可以理解成：

> 输出序列空间中的“位置/身份嵌入”。

---

## 8. query embeddings 和自回归动作生成有什么不同

ACT 不像传统自回归策略那样：

- 先生成 `a_1`
- 再把 `a_1` 喂回去生成 `a_2`
- 再继续生成 `a_3`

ACT 的做法是：

- 一次性放入 K 个 query
- 并行生成整个动作 chunk

因此 query embeddings 提供的是**并行输出槽位机制**，而不是“前一步动作历史”。

这带来两个重要区别。

### 8.1 优点：并行预测，效率高

因为一次就输出整个 chunk，  
推理时不用逐步 roll out K 次 decoder。

---

### 8.2 代价：需要靠 query + self-attention 建立槽位关系

既然不是自回归，就不能天然依赖“前一步动作已知”。  
所以模型必须用：

- query embeddings 区分各个输出位置
- decoder self-attention 建立这些位置之间的依赖关系

这也是为什么 query embeddings 在 ACT 中非常关键。

---

## 9. query embeddings 在 ACT 整体数据流中的位置

把它放回总流程里看，会更清楚。

1. 输入图像经过视觉 backbone，得到 [image tokens](image-tokens.md)
2. 当前机器人状态投影成 [proprio token](proprio-token.md)
3. 训练时由 posterior 采样 `z`，再投影成 [latent token](latent-token.md)
4. 这些 token 一起进入 encoder
5. encoder 输出统一的 [memory](memory.md)
6. decoder 读入零初始化 `tgt` 和 learned query embeddings
7. query 通过 self-attention / cross-attention 逐步变成动作槽位表示
8. 经 [action head](action-head.md) 输出动作 chunk

所以 query embeddings 的位置非常明确：

> 它不在输入端，不在 CVAE 端，而是在主 transformer 的 decoder 端，承担“输出槽位索引”的作用。

---

## 10. 常见误解

### 10.1 query embeddings 不是动作 token

它们不是“真实动作嵌入”，也不是 teacher forcing 输入。  
它们只是 decoder 的一组可学习查询位。

---

### 10.2 `tgt=0` 不代表 decoder 没有输入

虽然 `tgt` 的内容初始化为 0，  
但 decoder 仍然有重要输入：

- query embeddings
- encoder memory

所以 decoder 并不是“空跑”。

---

### 10.3 一个 query 不是只看一个输入 token

一个 query 在 cross-attention 中，通常会对整个 memory 加权读取。  
它不是与某个单独 patch 一一对应的。

---

### 10.4 query 对应动作槽位，不等于它只编码时间

动作槽位与 chunk 内顺序有关，但 query 学到的内容不只是“第几步”。  
它还会隐式学到：

- 该步常见动作模式
- 与前后槽位的协调关系
- 在不同观测条件下应关注什么信息

---

## 11. 一句话总结

我认为 query embeddings 可以这样记：

> query embeddings 是 decoder 里的可学习输出槽位，  
> 每个 query 对应 action chunk 的一个位置；  
> 它们以零初始化 `tgt` 为起点，通过 self-attention 协调槽位关系，再通过 cross-attention 从 encoder memory 中读取观测信息，最终变成可由 [action head](action-head.md) 映射为动作的隐藏表示。

---

## 12. 建议联动阅读

当前文件路径：`content/RoboTwin/ACT/concepts/query-embeddings.md`

因此下面这些链接都按当前目录相对路径来写：

- [ACT 整体数据流](../act-overall-dataflow.md)
- [训练与推理流程](../train-and-infer.md)
- [DETRVAE](../detr_vae.md)
- [latent token](latent-token.md)
- [proprio token](proprio-token.md)
- [image tokens](image-tokens.md)
- [memory](memory.md)
- [action head](action-head.md)
