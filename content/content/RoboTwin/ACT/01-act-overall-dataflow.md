---
title: ACT 整体数据流
---

# ACT 整体数据流

## 1. 这篇笔记要解决的问题

这篇笔记的目标不是只看一个文件，而是把 ACT 从输入到输出的整体前向过程串起来，回答以下问题：

1. ACT 的输入到底有哪些
2. 训练时和推理时的数据流有什么不同
3. latent token 从哪里来
4. 图像和机器人状态如何进入 transformer
5. decoder 如何输出动作 chunk
6. loss 如何反过来约束整个网络

这篇笔记相当于 ACT 的总览图。
如果只看单个模块，很容易知道“某段代码在干什么”，但不知道“它在整体链路里的位置”。

---

## 2. 按模块看，ACT 可以拆成哪几部分

在这份 RoboTwin 实现里，ACT 可以拆成七部分：

1. 输入部分
   - 图像 `image`
   - 机器人状态 `qpos`
   - 动作序列 `actions`（仅训练时有）
   - padding 标记 `is_pad`（仅训练时有）

2. posterior / latent 路径
   - 训练时，`qpos + actions` 进入 latent encoder
   - 得到 `mu`、`logvar`
   - 重参数化采样得到 `z`
   - 投影成 [[RoboTwin/ACT/concepts/01-latent-token|latent token]]

3. 视觉路径
   - 多路图像进入共享 backbone
   - 输出 feature map 和位置编码
   - 经过 `input_proj` 形成 [[RoboTwin/ACT/concepts/03-image-tokens|image tokens]]

4. 本体状态路径
   - `qpos` 经过线性投影
   - 得到 [[RoboTwin/ACT/concepts/02-proprio-token|proprio token]]

5. 主 transformer
   - encoder 输入由 latent token、proprio token、image tokens 组成
   - 输出整段 [[RoboTwin/ACT/concepts/04-memory|memory]]
   - decoder 使用 [[RoboTwin/ACT/concepts/05-query-embeddings|query embeddings]] 从 memory 中提取动作信息

6. 输出头
   - decoder 输出隐藏表示 `hs`
   - 经过 [[RoboTwin/ACT/concepts/06-action-head|action head]] 输出动作 chunk
   - 另外还定义了 `is_pad_head`

7. 损失
   - 动作 L1
   - KL 正则

---

## 3. 输入到底有哪些

### 3.1 图像 `image`

来自一个或多个相机。
在这份实现里，forward 里会按相机循环处理：

```python
for cam_id, cam_name in enumerate(self.camera_names):
    features, pos = self.backbones[0](image[:, cam_id])
```

说明多相机共用同一个 backbone 权重。

### 3.2 机器人状态 `qpos`

表示当前机器人本体状态，例如关节角等。
它一方面会进入 posterior encoder，另一方面也会投影成 [[RoboTwin/ACT/concepts/02-proprio-token|proprio token]] 进入主 encoder。

### 3.3 动作序列 `actions`

只在训练时存在。
它不会送进主 decoder，而是只用于构造 posterior，帮助模型学出 latent 条件。

### 3.4 `is_pad`

用于标记动作序列哪些位置是 padding。
它在两处起作用：

1. posterior encoder 中作为 action 序列 mask
2. 外层 L1 计算中屏蔽 padding 位置

### 3.5 这些输入到底从哪里来

如果你已经知道 ACT 的模型主体，但一到 `image / qpos / actions / is_pad` 的来源就容易混乱，那么建议把这一段和 [[RoboTwin/ACT/03-dataset-and-dataloader|Dataset 与 Dataloader 数据流]] 对照着看。

那一篇会专门讲清楚：

- `utils.py` 里的 `EpisodicDataset` 如何构造单样本
- 图像为什么是“单时刻多相机”
- `actions` 为什么是未来动作后缀
- `is_pad` 为什么必须和 `max_action_len` 一起出现
- DataLoader 为什么可以默认 collate 成 batch

---

## 4. 训练时的数据流

训练时的数据流是 ACT 最完整的版本，因为它同时包含：

- posterior 构造
- 主 transformer 动作预测
- loss 计算

### 4.1 第一步：构造 posterior

训练时，`qpos` 和真实动作 `actions` 会进入 latent encoder。
输入序列是：

```text
[CLS, qpos, a_1, a_2, ..., a_T]
```

然后：

- 取 `CLS` 位置输出
- 经过 `latent_proj`
- 切成 `mu` 和 `logvar`
- 重参数化采样得到 `z`
- 再投影成 [[RoboTwin/ACT/concepts/01-latent-token|latent token]]

这条路径的作用是：

> 把当前专家动作序列对应的“动作模式”压缩成一个低维潜变量条件。

注意：这条 posterior 路径**不看图像**。

---

### 4.2 第二步：提取图像特征

每路相机图像经过 backbone 后得到 feature map，随后再经过 `input_proj` 统一到 hidden dim。

如果有多路相机，这份实现会把它们：

- 沿宽度维拼接 feature map
- 沿宽度维拼接位置编码

即：

```python
src = torch.cat(all_cam_features, axis=3)
pos = torch.cat(all_cam_pos, axis=3)
```

然后才交给 transformer 统一 flatten。

所以更准确地说，图像不是“先手工变成一个个 token 再拼接”，而是：

> 先在 feature map 层面把多相机折叠进同一张宽图，再由 transformer 展平成 image token 序列。

---

### 4.3 第三步：构造 proprio token

当前机器人状态 `qpos` 经过：

```python
proprio_input = self.input_proj_robot_state(qpos)
```

得到一个 hidden-dim 向量，这就是 [[RoboTwin/ACT/concepts/02-proprio-token|proprio token]]。

---

### 4.4 第四步：送入主 encoder

在 `transformer.py` 里，图像 feature map flatten 后，会把：

- latent token
- proprio token

作为两个额外 token prepend 到图像序列前面。

所以主 encoder 看到的序列是：

```text
[latent, proprio, image tokens...]
```

这一步的含义是：

> 图像、状态、latent 条件不是最后才拼起来，而是一开始就共同参加 encoder self-attention 融合。

encoder 输出的不是一个 pooled 向量，而是整段 [[RoboTwin/ACT/concepts/04-memory|memory]]。

---

### 4.5 第五步：decoder 用 query 提取动作信息

decoder 输入不是历史动作，而是：

- 零初始化的 `tgt`
- learned query embeddings
- encoder 输出的 memory

其中：

- `self.query_embed = nn.Embedding(num_queries, hidden_dim)`
- `num_queries == chunk_size`

所以每个 query 都对应一个未来动作槽位。

decoder 通过：

1. query-query self-attention
2. query-memory cross-attention
3. FFN

逐步把 query 变成未来动作位置对应的 hidden state `hs`。

---

### 4.6 第六步：输出动作 chunk

decoder 输出 `hs` 后，模型直接做：

```python
a_hat = self.action_head(hs)
is_pad_hat = self.is_pad_head(hs)
```

于是得到：

- `a_hat`：预测动作 chunk
- `is_pad_hat`：padding 头输出

这里的关键理解是：

> 主要动作建模已经在 transformer 内部完成，action head 本身只是一个轻量线性映射头。

---

### 4.7 第七步：计算损失

在外层 `ACTPolicy` 中，训练时最终使用的是两部分 loss。

#### 1）动作 L1

```python
all_l1 = F.l1_loss(actions, a_hat, reduction="none")
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
```

这表示：

- 逐元素计算动作误差
- 再用 `is_pad` 屏蔽无效位置

#### 2）KL

由 `mu` 和 `logvar` 计算 posterior 相对标准高斯的 KL。

#### 总损失

```python
loss = l1 + kl * kl_weight
```

所以：

- `L1` 约束动作预测要准
- `KL` 约束 latent 空间要规整

注意：虽然模型里定义了 `is_pad_head`，但当前 `act_policy.py` **没有把 `is_pad_hat` 单独加入总损失**。

---

## 5. 推理时的数据流

推理时最大的变化只有一条：

> 没有真实未来动作，所以 posterior 分支不再存在。

于是模型内部直接：

```python
latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
latent_input = self.latent_out_proj(latent_sample)
```

也就是直接使用：

- `z = 0`
- 再投影成 latent token

除此之外，其余路径基本不变：

- 图像 backbone 一样
- proprio token 一样
- encoder 一样
- decoder query 一样
- action head 一样

所以推理可以概括成：

```text
z=0 -> latent token
image -> image tokens
qpos -> proprio token
[latent, proprio, image tokens] -> encoder -> memory
queries -> decoder(memory) -> hs -> action head -> a_hat
```

---

## 6. 为什么推理时直接 `z=0` 还能工作

因为训练时有 KL 项：

\[
D_{KL}(q(z|qpos, actions) \parallel \mathcal{N}(0, I))
\]

它会把训练时 posterior 压向标准高斯附近。

于是推理时即便没有真实动作，直接站在 prior 中心点 `z=0`，通常也仍能得到一个合理 latent 条件。

所以更准确地说：

> 不是“推理时随便给个 z 都行”，而是训练时已经通过 KL 把 latent 空间整理成了一个可以用默认中心点解码的空间。

---

## 7. 这条数据流如何真正落到环境动作上

在部署时，模型一次前向会输出整个 chunk。
外层 `ACT.get_action()` 会决定如何使用它。

### 不开 temporal aggregation

- 每隔 `num_queries` 步查询一次策略
- 从 chunk 中按当前位置依次取动作

### 开 temporal aggregation

- 每一步都重新预测一个 chunk
- 收集多个 chunk 对当前时刻的建议动作
- 再做指数加权平均

所以 ACT 的“chunk 输出”并不是直接一股脑全执行完，而是由外层 policy 决定如何在时间上消费这段输出。

---

## 8. 复习时最该立刻想起来的 8 个点

1. 训练时比推理时多了一条 posterior 分支
2. posterior 只看 `qpos + actions`，不看图像
3. latent token 是 `z` 投影后的条件 token
4. 多相机图像先在 feature map 层面沿宽度维拼接
5. 主 encoder 输入是 `[latent, proprio, image tokens...]`
6. decoder 不是喂真实动作，而是喂零初始化 `tgt + query embeddings`
7. `num_queries == chunk_size`，所以一个 query 对应一个动作槽位
8. 总损失是 `L1 + kl_weight * KL`

---

## 9. 一句话总结

> RoboTwin 里的 ACT 整体数据流可以概括为：训练时先用 `qpos + actions` 构造 posterior 得到 latent，再与图像和 qpos 一起进入主 transformer，最后通过 query 解码整个动作 chunk；推理时则省去 posterior，直接用 `z=0` 生成 latent token，其余主干保持不变。
