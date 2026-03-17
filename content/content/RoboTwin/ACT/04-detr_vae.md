---
title: DETR_VAE 结构
---

# DETR_VAE 结构

## 1. 这篇笔记要解决的问题

`policy/ACT/detr/models/detr_vae.py` 是这份 ACT 实现的核心模型文件。

它最容易让人混淆的点有两个：

1. 为什么文件名叫 `detr_vae`，但里面既有 VAE，又有 DETR 风格 transformer
2. latent encoder、视觉 backbone、主 transformer、action head 到底谁负责什么

所以这篇笔记的目标是：

> 把 `DETRVAE` 这个模型类内部的各个子模块拆开，并重新串成一条完整主线。

---

## 2. 先给总图：DETRVAE 里其实有四块东西

这份模型可以拆成四个子系统：

1. **latent encoder（CVAE 分支）**
   - 只在训练时使用真实动作去构造 posterior
2. **视觉 backbone**
   - 负责从多路相机图像提取 feature map
3. **主 transformer（encoder + decoder）**
   - 负责融合条件并用 query 解码动作
4. **输出头**
   - 把 decoder hidden state 映射成动作

所以 `DETRVAE` 不是“一个纯 VAE 模型”，而是：

> 一个“CVAE 条件分支 + DETR 风格主 transformer 动作解码器”的组合体。

---

## 3. 模型里最值得先记住的几个成员

在 `DETRVAE.__init__` 里，最关键的成员有：

```python
self.action_head = nn.Linear(hidden_dim, state_dim)
self.is_pad_head = nn.Linear(hidden_dim, 1)
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

这三者分别表示：

- `action_head`：把 decoder 输出映射到动作维度
- `is_pad_head`：预测每个 query 位置是否为 padding
- `query_embed`：decoder 端的 learned action slots

然后是 latent 相关：

```python
self.latent_dim = 32
self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
self.additional_pos_embed = nn.Embedding(2, hidden_dim)
```

这说明：

- 潜变量 `z` 维度是 32
- latent encoder 输出先被切成 `mu` 和 `logvar`
- 采样后的 `z` 再投影回 transformer hidden dim
- encoder 前面额外拼进去的两个 token（latent / proprio）还有各自 learned pos embed

---

## 4. latent encoder：它负责什么

### 4.1 它不是主 transformer encoder

这是最容易混淆的地方。

在这份实现里有两个“encoder”概念：

#### 第一类：latent encoder

- 作用：从 `qpos + actions` 构造 posterior
- 只在训练时起作用
- 用来得到 `mu`、`logvar` 和 latent sample

#### 第二类：主 transformer encoder

- 作用：融合 latent token、proprio token、image tokens
- 训练和推理都要用
- 输出给 decoder 查询的 [[RoboTwin/ACT/concepts/04-memory|memory]]

所以一定要把这两个 encoder 分开记。

> latent encoder 是 CVAE 分支的一部分；主 transformer encoder 才是动作生成主干的一部分。

---

### 4.2 latent encoder 的输入是什么

训练时，`DETRVAE.forward()` 会先做：

1. `actions -> encoder_action_proj`
2. `qpos -> encoder_joint_proj`
3. 拼接 `CLS token`

得到序列：

```text
[CLS, qpos, a_1, a_2, ..., a_T]
```

再配合 `is_pad` 做 mask 后送入 latent encoder。

这里有一个非常重要的实现细节：

> 这条 posterior 分支不看图像，它只看 `qpos + actions`。

这和很多人直觉上“latent 也应该看图像”不同。

---

### 4.3 latent encoder 的输出是什么

经过 latent encoder 后，取 `CLS` 位置的输出：

```python
encoder_output = encoder_output[0]
latent_info = self.latent_proj(encoder_output)
mu = latent_info[:, :self.latent_dim]
logvar = latent_info[:, self.latent_dim:]
```

于是得到：

- `mu`
- `logvar`

再重参数化采样得到 `latent_sample`，再经过：

```python
latent_input = self.latent_out_proj(latent_sample)
```

得到主 transformer 要使用的 [[RoboTwin/ACT/concepts/01-latent-token|latent token]]。

---

## 5. 视觉 backbone：它负责什么

这份实现里，视觉 backbone 的职责非常纯粹：

> 把多路相机图像变成高层 feature map 和对应位置编码。

在 forward 中，对每路相机都会做：

```python
features, pos = self.backbones[0](image[:, cam_id])
features = features[0]
pos = pos[0]
all_cam_features.append(self.input_proj(features))
all_cam_pos.append(pos)
```

注意两个点：

### 5.1 多路相机共享同一个 backbone 实例

代码里使用的是：

```python
self.backbones[0]
```

说明这份实现并不是每个相机一个独立 backbone，而是多相机共用一个 backbone 权重。

### 5.2 图像特征不会直接输出动作

backbone 只负责提特征，不负责动作回归。
它提供的是 [[RoboTwin/ACT/concepts/03-image-tokens|image tokens]] 的原材料。

真正的动作生成仍在后面的主 transformer 中完成。

---

## 6. 主 transformer：它负责什么

主 transformer 是整条动作生成主线的核心。

它接收三类条件：

1. [[RoboTwin/ACT/concepts/01-latent-token|latent token]]
2. [[RoboTwin/ACT/concepts/02-proprio-token|proprio token]]
3. [[RoboTwin/ACT/concepts/03-image-tokens|image tokens]]

然后分两步工作：

### 6.1 encoder：融合条件

在 `transformer.py` 中，图像 feature map 会先 flatten 成序列，再把：

- latent_input
- proprio_input

stack 成两个额外 token，prepend 到图像序列前面。

即：

```text
[latent, proprio, image tokens...]
```

然后 encoder 通过 self-attention 完成融合，输出整段 [[RoboTwin/ACT/concepts/04-memory|memory]]。

### 6.2 decoder：用 query 解码动作

decoder 端使用：

- `tgt = 0`
- `self.query_embed.weight`
- encoder memory

其中 query 个数就是 `num_queries`，而它在构建模型时等于 `chunk_size`。

所以 decoder 干的事是：

> 用一组 learned action slots 去 memory 中读取对应未来动作位置的信息。

详细可见：[[RoboTwin/ACT/concepts/05-query-embeddings|query embeddings]]。

---

## 7. action head：它负责什么

当 decoder 得到 `hs` 后，模型直接做：

```python
a_hat = self.action_head(hs)
is_pad_hat = self.is_pad_head(hs)
```

这说明：

- `hs` 已经是动作回归前最后一层表示
- `action_head` 很轻，本质上只是线性映射到动作维度
- `is_pad_head` 也是一个并行的轻头

也就是说，这份实现里真正“难”的建模工作不在 action head，而在：

- latent encoder 如何形成条件
- 主 encoder 如何融合多模态信息
- decoder 如何把 query 变成动作位表示

---

## 8. 训练时 forward 到底怎么串起来

训练时 forward 可以完整写成：

### 第一步：posterior 分支

```text
qpos + actions -> latent encoder -> mu, logvar -> reparameterize -> z -> latent token
```

### 第二步：视觉分支

```text
multi-camera images -> shared backbone -> feature maps -> input_proj -> image tokens
```

### 第三步：状态分支

```text
qpos -> input_proj_robot_state -> proprio token
```

### 第四步：主 transformer

```text
[latent, proprio, image tokens] -> encoder -> memory
queries -> decoder(memory) -> hs
```

### 第五步：输出头

```text
hs -> action_head -> a_hat
hs -> is_pad_head -> is_pad_hat
```

### 第六步：训练信号

```text
(mu, logvar) -> KL
(a_hat, actions) -> L1
```

---

## 9. 推理时 forward 又怎么变

推理时唯一被删掉的是 posterior 分支里的“真实动作输入”。

也就是：

- 没有 `actions`
- 没有 posterior encoder 输入
- 没有 `mu/logvar` 的有效计算
- 直接 `latent_sample = 0`

于是推理流程变成：

```text
z=0 -> latent token
multi-camera images -> backbone -> image tokens
qpos -> proprio token
[latent, proprio, image tokens] -> encoder -> memory
queries -> decoder(memory) -> hs -> action_head -> a_hat
```

所以你应该把 `DETRVAE` 记成：

> 一个训练时带 posterior 条件分支、推理时直接用 prior 中心点的 chunk 动作生成器。

---

## 10. 最容易混淆的几个点

### 10.1 `DETRVAE` 不是“只有一个 encoder”

它至少有：

- latent encoder
- 主 transformer encoder

二者作用完全不同。

### 10.2 CVAE 不等于视觉 backbone

CVAE 分支负责的是 latent 条件建模，
backbone 负责的是图像特征提取，二者不是一回事。

### 10.3 action head 不是主要建模模块

它只负责线性投影；真正的动作语义主要已经包含在 decoder 输出 `hs` 中。

### 10.4 `is_pad_head` 虽然存在，但当前训练外层并没有单独给它损失

这点在 [[RoboTwin/ACT/06-act_policy|act_policy]] 里要一起记住。

---

## 11. 一句话总结

> `DETRVAE` 是 RoboTwin 中 ACT 的核心模型壳：它把训练时的 CVAE posterior 条件分支、视觉 backbone、多模态主 transformer 和轻量 action head 组合起来，用于一次性预测一个动作 chunk。
