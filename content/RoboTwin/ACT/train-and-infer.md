---
title: 训练与推理流程
---

# 训练与推理流程

## 1. 这篇笔记要解决的问题

这篇笔记专门回答四个问题：

1. 训练时 ACT 的完整数据流到底怎么走
2. 推理时和训练时相比，少了哪条分支
3. 为什么训练时会有 posterior，而推理时直接用 `z=0`
4. 外层 policy 和 rollout 又是怎样使用模型输出的

如果说 [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]] 是总览页，
那么这篇更像是“训练版前向”和“部署版前向”的对照表。

---

## 2. 先给结论：训练和推理最大的区别是什么

在这份 RoboTwin 实现里，训练和推理的最大区别只有一条：

> **latent 的来源不同。**

更具体地说：

### 训练时

- 有真实未来动作 `actions`
- 可以构造 posterior encoder
- 得到 `mu`、`logvar`
- 重参数化采样得到 `z`
- 再投影成 [[RoboTwin/ACT/concepts/latent-token|latent token]]

### 推理时

- 没有真实未来动作
- 不能构造 posterior
- 直接令 `z = 0`
- 再投影成 latent token

除此之外，主干路径基本相同：

- 图像 backbone 一样
- proprio 投影一样
- 主 transformer 一样
- decoder query 一样
- [[RoboTwin/ACT/concepts/action-head|action head]] 一样

所以你可以把 ACT 记成：

> 训练和推理共享同一个“观测条件 → chunk 动作”的主网络，差别主要在 latent 条件是怎么来的。

---

## 3. 训练时的完整数据流

训练入口在：

- `policy/ACT/act_policy.py`
- `ACTPolicy.__call__(qpos, image, actions=None, is_pad=None)`

当 `actions is not None` 时，就走训练分支。

### 3.1 外层先做的事

训练时外层 policy 先做三件事：

#### 1）图像归一化

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image = normalize(image)
```

#### 2）把动作裁成 `num_queries` 长度

```python
actions = actions[:, :self.model.num_queries]
is_pad = is_pad[:, :self.model.num_queries]
```

这一步很关键。

因为模型一次只预测 `num_queries` 个动作位，而在本仓库里：

- `self.model.num_queries == chunk_size`

所以训练监督只保留当前 chunk 对应的那一段动作。

#### 3）把张量送进模型

```python
a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
```

这里 `env_state = None`，说明这份 RoboTwin 实现的 ACT 主路径主要依赖：

- 图像
- qpos
- actions（训练时只给 posterior encoder 用）

---

### 3.2 模型内部第一步：构造 posterior

在 `detr_vae.py` 里，训练时因为 `actions is not None`，所以会先进入 latent encoder 分支。

构造过程是：

1. `actions` 经过 `encoder_action_proj`
2. `qpos` 经过 `encoder_joint_proj`
3. 在最前面拼一个 `CLS token`
4. 得到序列：

```text
[CLS, qpos, a_1, a_2, ..., a_T]
```

5. 加位置编码和 padding mask
6. 送入 latent encoder
7. 取第一个位置，也就是 `CLS` 对应输出
8. 经过 `latent_proj` 切成 `mu` 和 `logvar`

所以训练时 posterior 表示的是：

> 在“当前状态 + 真实未来动作”条件下，这条动作序列对应的潜变量分布。

这就是为什么它叫：

\[
q(z \mid qpos, actions)
\]

注意，这里**没有图像进入 latent encoder**。这是你复习时一定要记住的实现细节。

---

### 3.3 模型内部第二步：重参数化得到 latent token

训练时接下来做：

\[
z = \mu + \sigma \odot \epsilon
\]

然后：

```python
latent_input = self.latent_out_proj(latent_sample)
```

这样就得到主 transformer 要用的 [[RoboTwin/ACT/concepts/latent-token|latent token]]。

它的作用不是直接输出动作，而是给后面的主网络提供“这条专家轨迹的动作模式条件”。

---

### 3.4 模型内部第三步：图像路径

在这份实现里，图像处理流程是：

1. 对每个相机分别调用同一个 backbone：

```python
features, pos = self.backbones[0](image[:, cam_id])
```

2. 取最后一层 feature map
3. 经过 `input_proj` 映射到 transformer hidden dim
4. 每路相机的特征图沿宽度维拼接：

```python
src = torch.cat(all_cam_features, axis=3)
pos = torch.cat(all_cam_pos, axis=3)
```

这说明这份实现里多相机不是先变成独立 token 列表再手工拼，而是：

> 先保留 2D feature map 结构，把 camera 维折叠进宽度维，再交给 transformer 统一 flatten。

---

### 3.5 模型内部第四步：qpos 路径

当前机器人状态 `qpos` 会经过：

```python
proprio_input = self.input_proj_robot_state(qpos)
```

得到 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]。

这个 token 和 latent token 一样，不是图像 patch，而是额外条件 token。

---

### 3.6 模型内部第五步：进入主 transformer

在 `transformer.py` 中，如果输入是图像 feature map，就会做这些事：

1. 把图像 feature map flatten 成序列
2. 把 `latent_input` 和 `proprio_input` stack 成两个额外 token
3. 把它们 prepend 到图像 token 序列前面：

```python
addition_input = torch.stack([latent_input, proprio_input], axis=0)
src = torch.cat([addition_input, src], axis=0)
```

4. 再把 `additional_pos_embed` 也拼到位置编码前面
5. 送入 encoder 得到 [[RoboTwin/ACT/concepts/memory|memory]]

所以这份实现里 encoder 真正看到的序列是：

```text
[latent, proprio, image tokens...]
```

这一步非常重要，因为它说明：

> latent 和 proprio 不是在最后阶段才拼接，而是一开始就作为 token 参与 encoder 融合。

---

### 3.7 模型内部第六步：decoder 生成动作 chunk

decoder 端会使用：

- 全 0 初始化的 `tgt`
- `self.query_embed.weight`
- encoder 输出的 `memory`

其中：

- `num_queries == chunk_size`
- 所以一个 query 对应 chunk 中一个动作位置

经过 decoder 之后，得到：

- `hs`：每个动作槽位的 hidden state

接着做：

```python
a_hat = self.action_head(hs)
is_pad_hat = self.is_pad_head(hs)
```

于是训练前向结束时会拿到：

- `a_hat`：预测动作 chunk
- `is_pad_hat`：padding 预测头输出
- `mu, logvar`：给 KL 用

---

### 3.8 训练时 loss 怎么算

外层 `ACTPolicy` 训练时只用了两部分 loss：

#### 1）动作 L1

```python
all_l1 = F.l1_loss(actions, a_hat, reduction="none")
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
```

这里表示：

- 先逐元素算 L1
- 再用 `~is_pad` 把 padding 位置屏蔽掉
- 最后求平均

所以 L1 真正约束的是：

> 非 padding 的有效动作位置上，预测动作和专家动作要接近。

#### 2）KL

```python
total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
```

其中最终加入 loss 的是：

```python
loss = l1 + kl * kl_weight
```

所以 KL 的作用是：

> 把训练时 posterior 压向标准高斯附近，使 latent 空间规整，保证推理时直接用 prior 中心点也能工作。

注意：虽然模型结构里定义了 `is_pad_head`，但当前 `act_policy.py` **没有把 `is_pad_hat` 单独加入训练损失**。

---

## 4. 推理时的数据流

推理入口仍然是 `ACTPolicy.__call__`，但这时 `actions is None`。

于是外层会直接调用：

```python
a_hat, _, (_, _) = self.model(qpos, image, env_state)
```

这时模型内部和训练时相比，唯一真正消失的分支就是 posterior encoder。

---

### 4.1 推理时不再有 posterior

因为没有真实未来动作，所以不能构造：

\[
q(z \mid qpos, actions)
\]

因此推理时在 `detr_vae.py` 中直接执行：

```python
latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
latent_input = self.latent_out_proj(latent_sample)
```

所以推理时不是从 posterior 采样，也不是随机从高斯里 sample，
而是直接使用：

> `z = 0` 这个 prior 中心点。

这是这份实现非常值得记住的地方。

---

### 4.2 为什么直接 `z=0` 可以工作

因为训练时有 KL 项：

\[
D_{KL}(q(z\mid qpos, actions) \parallel \mathcal{N}(0, I))
\]

它会强迫 posterior 不要偏离标准高斯太远。

于是推理时即便不用真实动作去推 posterior，直接使用 prior 的中心点 `0`，
通常也仍能落在一个合理的 latent 区域里。

可以把它记成：

> 训练时靠 KL 把 latent 空间“拢”到标准高斯附近，推理时就能直接站在中心点 `z=0` 来解码动作。

---

### 4.3 推理时其余主干不变

除了 latent 来源变了，下面这些都不变：

- 图像 backbone 一样
- 多相机特征拼接方式一样
- proprio 投影一样
- encoder 融合一样
- decoder query 解码一样
- action head 回归一样

所以你在脑中最好形成这样一个对照：

### 训练时

```text
(qpos, actions) -> posterior -> z -> latent token
image -> backbone -> image tokens
qpos -> proprio token
[latent, proprio, image tokens] -> encoder -> memory
queries -> decoder(memory) -> hs -> action head -> a_hat
                             \-> KL(mu, logvar)
```

### 推理时

```text
z=0 -> latent token
image -> backbone -> image tokens
qpos -> proprio token
[latent, proprio, image tokens] -> encoder -> memory
queries -> decoder(memory) -> hs -> action head -> a_hat
```

---

## 5. rollout / 部署时又是怎么用这个 chunk 的

只理解 `a_hat` 是一串未来动作还不够，还要知道外层部署怎么使用它。

在 `policy/ACT/act_policy.py` 的 `ACT.get_action()` 里，部署逻辑是：

### 情况 A：不开 temporal aggregation

- 每隔 `query_frequency = num_queries` 步查询一次策略
- 得到一个长度为 `num_queries` 的动作 chunk
- 当前时刻取：

```python
raw_action = self.all_actions[:, self.t % self.query_frequency]
```

也就是说：

> 一次前向预测一整段，然后每个环境步从这段里按位置依次取出一个动作执行。

### 情况 B：开 temporal aggregation

如果 `temporal_agg=True`，那么：

- `query_frequency = 1`
- 每个环境步都重新预测一个 chunk
- 所有历史 chunk 对当前时刻的候选动作都会被收集起来
- 再按指数权重做加权平均

也就是说：

> 当前时刻的最终动作，不一定只来自最新一次预测，而是多个 chunk 对该时刻建议动作的加权融合。

这部分不属于 ACT 主网络结构本身，但属于这份 RoboTwin 实现里“动作 chunk 怎么真正落地”的关键。

---

## 6. 训练与推理最应该对照记住的点

### 相同点

1. 图像都经过同一个视觉 backbone
2. qpos 都会投影成 proprio token
3. 主 encoder / decoder 完全共用
4. query embeddings 完全相同
5. action head 完全相同

### 不同点

1. 训练时有 `actions`，推理时没有
2. 训练时会构造 posterior encoder，推理时不会
3. 训练时 latent 来自 `q(z|qpos, actions)`，推理时 latent 直接取 `z=0`
4. 训练时有 L1 + KL，推理时不算损失

---

## 7. 一句话总结

> 在 RoboTwin 这份 ACT 实现里，训练和推理共享同一个主 transformer 动作生成路径；
> 真正的区别主要在于训练时用 `qpos + actions` 构造 posterior 得到 latent，而推理时没有真实动作，所以直接用 `z=0` 生成 latent token，再解码出整个动作 chunk。
