---
title: memory
---

# memory

## 1. 这篇笔记要回答什么

在 ACT 里，`memory` 这个词很容易被说得很抽象，好像只是“transformer 的记忆库”。

但在你这份 RoboTwin 仓库实现里，`memory` 其实非常具体：

> 它就是 **主 transformer encoder 输出的整段 token 序列**，
> 里面已经融合了 [[RoboTwin/ACT/concepts/01-latent-token|latent token]]、[[RoboTwin/ACT/concepts/02-proprio-token|proprio token]] 和 [[RoboTwin/ACT/concepts/03-image-tokens|image tokens]] 的信息。

decoder 后续做的事，不是凭空生成动作，而是让 [[RoboTwin/ACT/concepts/05-query-embeddings|query embeddings]] 去这段 encoder 输出序列里读信息，最后由 [[RoboTwin/ACT/concepts/06-action-head|action head]] 映射成动作。

所以理解 `memory` 的关键不是把它当成“某个额外模块”，而是明白：

- 它来自哪里
- 它里面装了什么
- decoder 为什么一定要读它

---

## 2. 在这份仓库里，memory 来自哪里

主线在 `policy/ACT/detr/models/detr_vae.py`：

1. 图像送入 backbone，得到每路相机的 feature map 和位置编码
2. `qpos` 经过线性层投影成 proprio token
3. 训练时，动作序列先经过 CVAE encoder 得到 latent sample，再投影成 latent token；推理时直接用零向量当 latent sample，再投影
4. 这些信息一起送进 `self.transformer(...)`
5. transformer encoder 输出一整段隐藏表示，这段输出就是 decoder cross-attention 要读取的 `memory`

也就是说，`memory` 不是额外存下来的缓存，而是：

> **encoder 处理完当前观测与条件信息之后得到的序列表征。**

---

## 3. memory 的组成：它不是单个向量，而是一整段序列

这是最容易误解的地方。

很多人第一次听到 memory，会下意识把它想成：

- 一个全局向量
- 一个 pooled feature
- 或者一个“摘要状态”

但在这份实现里，并不是这样。

decoder cross-attention 读取的不是单个 pooled 向量，而是 **encoder 输出的整个 token 序列**。因此 memory 里保留了细粒度结构。

从概念上，可以把 encoder 输入写成：

```text
[latent token, proprio token, image tokens...]
```

经过 encoder 后，对应得到：

```text
[memory_latent, memory_proprio, memory_img_1, memory_img_2, ...]
```

这里每一个位置都不再是原始输入 token，而是：

- 已经过多层 self-attention 融合后的隐藏表示
- 同时携带自身信息和其他 token 的上下文信息

所以 `memory` 最准确的理解是：

> 一串“上下文增强后的条件表示”，而不是某个单点摘要。

---

## 4. 为什么 memory 必须由 encoder 先融合一遍

如果没有 encoder，decoder 也可以直接对图像 token、qpos、latent 做 cross-attention。那为什么 ACT 不这么做？

原因在于：

### 4.1 encoder 先把多模态条件对齐到同一个隐藏空间

在这份实现里，输入端至少有三类信息：

- latent：高层动作模式条件
- proprio：当前机器人关节状态
- image：视觉观测

它们来源不同、语义不同。

encoder 的第一层作用，就是把这些不同来源的 token 放到统一的 token 空间里，让它们先彼此交流。

---

### 4.2 encoder 先把“条件之间的关系”建模好

decoder 需要知道的不是单独的图像特征，而是类似下面这种融合关系：

- 当前手臂状态和目标物体位置关系如何
- latent 规定的动作风格，与当前视觉状态是否匹配
- 哪些图像区域与当前 proprio 状态最相关

这些关系如果不先在 encoder 里融合，decoder 每个 query 都得自己从零去解决条件对齐问题，效率会更差。

因此 encoder 的工作可以理解为：

> 先把“当前场景条件”整理成一个统一、可检索的 memory，再交给 decoder 按动作槽位去读取。

---

## 5. 在 RoboTwin 实现里，memory 是怎么形成的

### 5.1 image tokens 的来源

每个相机都经过同一个 backbone：

- `features, pos = self.backbones[0](image[:, cam_id])`
- `features = self.input_proj(features[0])`

然后多路相机特征在宽度维拼接：

```text
src = cat(all_cam_features, axis=3)
pos = cat(all_cam_pos, axis=3)
```

这意味着：

- 多路相机没有先独立过 transformer encoder
- 而是先变成一个更宽的 feature map，再统一送入主 transformer

图像真正变成 token 序列，是在 `transformer.py` 里 flatten 之后发生的。

---

### 5.2 proprio token 的来源

`qpos` 经过：

```python
self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
```

得到一个 hidden_dim 维向量，作为 proprio token。

它不是 patch，也不是时序序列，而是一个单独的状态 token。

见：[[RoboTwin/ACT/concepts/02-proprio-token|proprio token]]

---

### 5.3 latent token 的来源

训练时：

- `actions` 和 `qpos` 进入 CVAE encoder
- 先得到 `mu, logvar`
- 再通过 reparameterization 采样 latent sample
- 最后用 `latent_out_proj` 投影到 hidden_dim

推理时：

- 不走 posterior encoder
- 直接用 `zeros([bs, latent_dim])`
- 再做同样的 `latent_out_proj`

见：[[RoboTwin/ACT/concepts/01-latent-token|latent token]]

---

### 5.4 在 transformer 中拼成 encoder 输入

RoboTwin 这份实现里，latent 和 proprio 不是和图像一样从 feature map flatten 来的，而是作为 `additional_input` 单独拼到 encoder 序列前面。

所以从语义上看，encoder 输入顺序就是：

```text
[latent token, proprio token, flattened image tokens...]
```

然后再一起经过 encoder，形成 `memory`。

---

## 6. memory 和原始 token 有什么不同

这个区别特别重要。

### 原始 token 只是“初始表示”

例如：

- latent token 初始只代表 latent sample 的投影
- proprio token 初始只代表 qpos 的投影
- image tokens 初始只代表某些局部视觉 patch 的表示

它们彼此之间还没有充分融合。

### memory 是“融合后的表示”

经过 encoder 多层 self-attention 之后，每个位置都已经吸收了上下文。

例如某个 image token 在 memory 中，已经不只是“某个 patch 长什么样”，而更可能同时带着：

- 当前机器人姿态信息
- latent 规定的动作模式信息
- 其他相机视角给出的补充上下文

所以可以说：

> token 是输入侧的局部条件表示，memory 是 encoder 输出侧的上下文增强表示。

---

## 7. decoder 是怎样使用 memory 的

这部分必须和 [[RoboTwin/ACT/concepts/05-query-embeddings|query embeddings]] 连起来理解。

decoder 一开始有：

- 零初始化的 `tgt`
- learned `query_embed.weight`

然后每层做两件事：

1. self-attention：让不同动作槽位之间先彼此协调
2. cross-attention：让每个 query 去读取 encoder memory

因此，对 decoder 来说，memory 的作用不是“历史缓存”，而是：

> 当前时刻所有条件信息的统一检索库。

每个 query 都可以从中读取：

- 与当前动作槽位最相关的视觉线索
- 当前机器人状态
- latent 给出的高层动作模式

这就是为什么 query 最后能变成动作表示。

---

## 8. 为什么说 memory 是 ACT 的“条件中心”

你可以把整套 ACT 主线分成两段：

### 第一段：把条件整理成 memory

输入：

- 图像
- qpos
- latent（训练时来自 posterior，推理时为零）

输出：

- encoder memory

### 第二段：从 memory 中抽取动作槽位表示

输入：

- query embeddings
- memory

输出：

- 每个动作槽位的 hidden state
- 再经 action head 变成动作 chunk

所以 memory 正好位于这两段之间，充当“条件中心枢纽”。

---

## 9. 和 CVAE encoder 不要混淆

这也是 ACT 里很容易混的点。

在你仓库里其实有两个“编码过程”：

### 9.1 CVAE encoder

位置：`detr_vae.py` 里 `self.encoder`

用途：

- 只在训练时使用
- 输入 `CLS + qpos + actions`
- 输出 `mu, logvar`
- 用来产生 latent sample

它不是给 decoder 提供 memory 的那个 encoder。

---

### 9.2 主 transformer encoder

位置：`self.transformer(...)` 内部

用途：

- 训练和推理都使用
- 输入 `latent token + proprio token + image tokens`
- 输出 memory
- 给 decoder cross-attention 读取

所以当你在 ACT 笔记里写 memory 时，默认指的是：

> **主 transformer encoder 的输出序列**，不是 posterior encoder 的输出。

---

## 10. 和源码的一一对应

你之后看源码时，可以直接这样对照：

### 在 `detr_vae.py`

- `latent_input`：latent token 的初始表示
- `proprio_input`：proprio token 的初始表示
- `src`：拼接后的多相机图像特征图
- `pos`：图像 token 的位置编码
- `self.transformer(...)`：主 transformer
- `hs = self.transformer(...)[0]`：decoder 输出 hidden states

虽然 `memory` 不在 `detr_vae.py` 里被单独命名返回给上层，但它确实在 `transformer.py` 的 encoder-decoder 流里存在，并被 decoder cross-attention 使用。

---

## 11. 复习时怎么记

我建议把 memory 记成这句话：

> memory 不是额外模块，而是主 transformer encoder 输出的整段融合后序列；
> 它把 latent、proprio、image 三类条件统一到一个可被 decoder 检索的表示空间里。

这样你以后看到：

- `query`
- `cross-attention`
- `decoder 读 memory`

就不会把它想成某种模糊“缓存”，而会立刻反应到：

- 它来自 encoder
- 它是一整段 token 序列
- 它承载了当前时刻所有动作生成条件

---

## 12. 一句话总结

> 在 RoboTwin 的 ACT 实现里，memory 就是主 transformer encoder 输出的上下文增强 token 序列；
> 它由 latent token、proprio token 和 image tokens 融合而来，随后被 query embeddings 通过 cross-attention 读取，用于生成动作 chunk。
