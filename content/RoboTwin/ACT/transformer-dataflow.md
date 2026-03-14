---
title: ACT 中 Transformer 数据流转
---

# ACT 中 Transformer 数据流转

## 1. 这篇笔记要解决的问题

这篇笔记不再只讲“通用版 ACT transformer 在做什么”，而是专门对应 RoboTwin 仓库里的实现，回答下面几个问题：

1. `policy/ACT/detr/models/detr_vae.py` 里，主 transformer 的输入到底由哪些部分组成
2. 图像特征、`latent_input`、`proprio_input` 是怎样一起进入 transformer 的
3. `query_embed`、零初始化 `tgt`、decoder 的 self-attention / cross-attention 各自负责什么
4. `hs` 是怎样变成动作 chunk 的
5. 这一页和 [[RoboTwin/ACT/detr_vae|detr_vae]]、[[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]、[[RoboTwin/ACT/concepts/memory|memory]] 的关系是什么

这篇的重点是：

> 把“ACT 主 transformer 从输入到输出的真实数据流”讲清楚，并尽量和仓库代码一一对应。

---

## 2. 先分清两个 encoder：不要混淆

在这份实现里，“encoder”其实有两层含义，必须先分开。

### 2.1 latent encoder / posterior encoder

对应 `DETRVAE.forward()` 里训练阶段的这部分：

- `encoder_action_proj(actions)`
- `encoder_joint_proj(qpos)`
- `self.encoder(...)`
- `latent_proj(...)`

它的作用是：

> 训练时把 `qpos + action sequence` 编成一个 latent 分布，再采样出 `z`。

这部分属于 CVAE 分支，不是我们这篇要重点讲的“主 transformer 编码观测再解码动作”的主干。

---

### 2.2 主 transformer

对应 `self.transformer(...)` 这一段。

它的作用是：

1. 把当前观测条件组织成 encoder memory
2. 再让 decoder 里的 query 去读取这段 memory
3. 最后输出动作 chunk 对应的 hidden states

所以这篇说的 **transformer 数据流**，主要是指这条主线：

```text
image / qpos / latent_input
    -> 主 transformer encoder
    -> memory
    -> 主 transformer decoder + query_embed
    -> hs
    -> action_head
    -> a_hat
```

---

## 3. 主 transformer 的输入由哪些部分组成

在 `detr_vae.py` 中，图像路径分支存在时，核心代码是：

```python
proprio_input = self.input_proj_robot_state(qpos)
src = torch.cat(all_cam_features, axis=3)
pos = torch.cat(all_cam_pos, axis=3)
hs = self.transformer(
    src,
    None,
    self.query_embed.weight,
    pos,
    latent_input,
    proprio_input,
    self.additional_pos_embed.weight,
)[0]
```

这说明主 transformer 实际接收了 4 类关键信息。

### 3.1 image features

来自多路相机图像：

1. 每个相机图像 `image[:, cam_id]`
2. 经 `self.backbones[0](...)` 提取 feature map 和 positional embedding
3. 再经 `self.input_proj` 投到 `hidden_dim`
4. 最后沿宽度维拼接成统一的 `src`

所以图像信息不是一开始就变成一维 token 列表，而是：

- 先保持 2D feature map 形式
- 多相机沿 width 维拼起来
- 再在 transformer 内部 flatten

这点和只看概念图时的直觉不太一样。

详见 [[RoboTwin/ACT/concepts/image-tokens|image tokens]]。

---

### 3.2 proprio_input

来自当前机器人状态 `qpos`：

```python
proprio_input = self.input_proj_robot_state(qpos)
```

也就是把机器人本体状态线性映射到 `hidden_dim`，作为一个额外条件 token。

详见 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]。

---

### 3.3 latent_input

来自 CVAE 的 latent 分支：

- 训练时：由 posterior encoder 从 `qpos + actions` 中得到 `mu, logvar`，再 reparameterize 得到 `z`
- 推理时：直接取零向量 `z=0`
- 之后统一经 `latent_out_proj` 投到 `hidden_dim`

```python
latent_input = self.latent_out_proj(latent_sample)
```

这个 `latent_input` 也会作为额外 token 加到主 transformer encoder 的输入前面。

详见 [[RoboTwin/ACT/concepts/latent-token|latent token]]。

---

### 3.4 query_embed

来自：

```python
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

这里的 `num_queries` 在构建模型时会被设为 `chunk_size`。所以在这份实现里：

> query 的个数 = 一次要并行预测的动作步数。

这组 query 不进入 encoder，而是进入 decoder，作为动作槽位。

详见 [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]。

---

## 4. encoder 端：观测条件是怎样变成 memory 的

### 4.1 多相机特征先沿宽度维拼接

在 `detr_vae.py` 中：

```python
all_cam_features.append(self.input_proj(features))
all_cam_pos.append(pos)
...
src = torch.cat(all_cam_features, axis=3)
pos = torch.cat(all_cam_pos, axis=3)
```

这一步的含义是：

- 每个相机得到一张 feature map
- 每个 feature map 都已经对齐到 `hidden_dim`
- 然后多相机在空间维上拼成一张更“宽”的特征图

所以主 transformer 看到的不是“相机 1 token 序列 + 相机 2 token 序列”这种显式列表，
而是一个已经在空间维上合并好的视觉源。

---

### 4.2 transformer 内部再把视觉特征 flatten 成 token 序列

在 `transformer.py` 里，开头会做：

```python
bs, c, h, w = src.shape
src = src.flatten(2).permute(2, 0, 1)
pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
```

这时视觉特征才真正变成一个 token 序列：

- 序列长度 = `h * w`
- 每个 token 维度 = `hidden_dim`

所以如果你以后要回答“image tokens 是在哪里形成的”，更准确的答案是：

> `detr_vae.py` 负责准备和拼接 feature map，`transformer.py` 里通过 `flatten(2).permute(2,0,1)` 真正把它变成 encoder token 序列。

---

### 4.3 latent / proprio 作为额外 token prepend 到视觉 token 前面

这是这份实现非常关键的一点。

`transformer.py` 里会做：

```python
additional_input = torch.stack([latent_input, proprio_input], axis=0)
src = torch.cat([additional_input, src], axis=0)
additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)
pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)
```

这意味着 encoder 输入序列实际是：

```text
[latent token, proprio token, image token_1, image token_2, ..., image token_N]
```

注意这里不是把 latent / proprio 和图像直接相加，而是：

- 把它们作为**独立 token**拼在视觉 token 前面
- 同时给这两个额外 token 配置单独的 `additional_pos_embed`

这也解释了为什么 [[RoboTwin/ACT/concepts/memory|memory]] 不是只有视觉信息，而是融合后的整段上下文化序列。

---

### 4.4 encoder 的输出就是 memory

接着：

```python
memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
```

这里的 `memory` 不是一个 pooled 向量，也不是单个全局特征，而是：

> encoder 输出的整段 token 序列。

它包含：

- latent token 更新后的表示
- proprio token 更新后的表示
- 所有 image tokens 更新后的表示

这些 token 在 self-attention 中已经彼此交互过，所以每个 token 都带有跨模态上下文。

---

## 5. decoder 端：query 是怎样把 memory 读成动作表示的

### 5.1 query 的来源

在 `detr_vae.py` 中：

```python
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

传入 transformer 时用的是：

```python
self.query_embed.weight
```

也就是说，decoder 的 query 不是从输入观测里算出来的，而是一组模型参数。

它们表示的是：

> “我要输出第 1 个动作槽位、第 2 个动作槽位、……、第 K 个动作槽位。”

其中 `K = num_queries = chunk_size`。

---

### 5.2 为什么 decoder 的 `tgt` 可以全零

在 `transformer.py` 中：

```python
tgt = torch.zeros_like(query_embed)
```

这一点很容易让人误会成：“decoder 没有输入”。

其实不是。

decoder 这里有两类输入：

1. `tgt`：当前槽位的内容表示，初始化为 0
2. `query_pos = query_embed`：当前槽位的身份 / 位置表示

所以零初始化 `tgt` 的含义是：

> 初始时每个输出槽位还没有内容，但每个槽位的身份已经通过 `query_embed` 区分开了。

因此 decoder 仍然能区分：

- 哪个槽位负责 chunk 第 1 步
- 哪个槽位负责 chunk 第 2 步
- 哪个槽位负责 chunk 第 3 步

---

### 5.3 self-attention：动作槽位之间先彼此协调

在 decoder layer 里，先做的是：

```python
q = k = self.with_pos_embed(tgt, query_pos)
tgt2 = self.self_attn(q, k, value=tgt, ...)[0]
```

这里参与 self-attention 的对象不是图像 token，而是 decoder 内部的各个 query 槽位。

它的作用是：

> 让 chunk 内不同动作位置先建立关系。

因为 action chunk 不是 K 个毫无关联的独立动作，而是一个连续动作片段。比如：

- 前一步是靠近
- 下一步是抓取
- 再下一步是抬起

这些输出位之间本来就有依赖关系，所以 self-attention 先让 query 之间交换信息。

---

### 5.4 cross-attention：每个 query 去 memory 里读自己需要的信息

随后 decoder 会做 cross-attention，大致形式是：

```python
tgt2 = self.multihead_attn(
    query=self.with_pos_embed(tgt, query_pos),
    key=self.with_pos_embed(memory, pos),
    value=memory,
    ...
)[0]
```

这一步最核心。

它表示：

- query 侧：当前动作槽位
- key/value 侧：encoder 输出的 memory

于是每个 query 会从 memory 里读取：

- 当前场景视觉信息
- 当前机器人姿态信息
- latent 提供的高层动作模式信息

最终把自己更新成：

> 在当前观测条件下，“第 i 个动作槽位应该输出什么”的隐藏表示。

---

### 5.5 多层 decoder 反复细化，得到 `hs`

每层 decoder 都会重复：

1. query-query self-attention
2. query-memory cross-attention
3. FFN

最后得到的 `hs` 可以理解为：

> 每个 query 对应的最终动作隐藏向量。

在这份实现中，transformer 返回后会直接做：

```python
a_hat = self.action_head(hs)
is_pad_hat = self.is_pad_head(hs)
```

这说明：

- 主要的建模工作已经在 transformer 中完成
- `action_head` 本身只是一个比较轻的线性映射头

详见 [[RoboTwin/ACT/concepts/action-head|action head]]。

---

## 6. `hs` 到动作输出：为什么说一个 query 对应一个动作槽位

在 `detr_vae.py` 中：

```python
a_hat = self.action_head(hs)
```

而 `hs` 的 query 维长度就是 `num_queries`。

因此输出天然就是：

```text
hs[0] -> 动作槽位 1
hs[1] -> 动作槽位 2
...
hs[K-1] -> 动作槽位 K
```

这不是“概念上大概如此”，而是这份实现直接决定的结构事实。

所以这里要牢牢记住：

> query 的个数和动作 chunk 的长度一一对应，decoder 输出的每个 query hidden state，都会被直接映射为 chunk 中对应位置的动作。

---

## 7. 把整条主 transformer 数据流写成一条线

如果按这份仓库实现，把主 transformer 单独抽出来，可以写成：

```text
多路相机图像
  -> backbone
  -> input_proj
  -> 多相机沿 width 维拼接
  -> transformer 内部 flatten 成 image tokens

qpos
  -> input_proj_robot_state
  -> proprio_input

训练时 actions + qpos
  -> posterior encoder
  -> z
  -> latent_out_proj
  -> latent_input

推理时
  -> z = 0
  -> latent_out_proj
  -> latent_input

[latent token, proprio token, image tokens]
  -> transformer encoder
  -> memory

query_embed + zero tgt
  -> transformer decoder
  -> hs
  -> action_head
  -> a_hat
```

这条线和 [[RoboTwin/ACT/train-and-infer|训练与推理流程]]、[[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]] 是一致的。

---

## 8. 这篇最容易和源码对上的几个位置

如果你以后回源码，最应该一眼对上的就是下面几处。

### 8.1 `detr_vae.py`

重点看：

- `self.query_embed = nn.Embedding(num_queries, hidden_dim)`
- `latent_out_proj`
- `input_proj_robot_state`
- 多相机 feature 拼接
- `self.transformer(...)`
- `a_hat = self.action_head(hs)`

这部分决定“输入有什么、输出是什么”。

---

### 8.2 `transformer.py`

重点看：

- `src.flatten(2).permute(2, 0, 1)`
- `torch.stack([latent_input, proprio_input], axis=0)`
- `torch.cat([additional_input, src], axis=0)`
- `tgt = torch.zeros_like(query_embed)`
- decoder layer 里的 `self_attn` 与 `multihead_attn`

这部分决定“中间是怎么流的”。

---

## 9. 复习时最应该记住的点

### 9.1 主 transformer 的 encoder 输入不只是图像

它实际接收：

- latent token
- proprio token
- image tokens

而且 latent / proprio 是作为额外 token prepend 到视觉 token 前面的。

---

### 9.2 image tokens 是在 transformer 内部 flatten 出来的

`detr_vae.py` 先准备 feature map，`transformer.py` 再真正 flatten 成 token 序列。

---

### 9.3 memory 是 encoder 输出的整段 token 序列

不是单个 pooled 向量，也不是抽象意义上的“记忆库”而已。

---

### 9.4 query 的个数就是 chunk size

这份实现里，一个 query 就对应一个动作槽位。

---

### 9.5 `tgt=0` 不等于 decoder 没输入

decoder 还有 `query_embed` 和 `memory`。

---

### 9.6 transformer 完成主要建模，action head 只是轻量映射

`hs -> action_head -> a_hat` 很直接。

---

## 10. 一句话总结

我认为 RoboTwin 这份 ACT 实现里的主 transformer，可以这样记：

> encoder 先把 `latent + proprio + 多相机视觉` 融合成 memory；
> decoder 再用和 chunk 长度一一对应的 learned queries，从这段 memory 中并行读取每个未来动作槽位所需的信息；
> 最后通过轻量 action head 直接输出整个动作 chunk。
