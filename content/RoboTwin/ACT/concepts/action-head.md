---
title: action head
---

# action head

## 1. action head 是什么

在 ACT 里，`action head` 是把 decoder 输出的每个动作槽位 hidden state，映射成实际动作向量的最后一层头部。

在你这份 RoboTwin 仓库实现里，它非常直接：

```python
self.action_head = nn.Linear(hidden_dim, state_dim)
```

也就是说，action head 不是复杂网络，而是一个简单线性层。  
因此它的含义可以概括为：

> 主 transformer 负责把动作相关信息建模好，action head 只负责把最终 hidden representation 读出来，映射到动作维度。

---

## 2. 它位于整条 ACT 数据流的哪里

ACT 主线可以粗略拆成：

1. 输入条件构造
   - [[RoboTwin/ACT/concepts/image-tokens|image tokens]]
   - [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
   - [[RoboTwin/ACT/concepts/latent-token|latent token]]
2. encoder 融合这些条件，形成 [[RoboTwin/ACT/concepts/memory|memory]]
3. decoder 用 [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]] 从 memory 中抽取每个动作槽位的表示
4. action head 把每个槽位表示映射成动作

所以 action head 的位置非常靠后。  
它不是负责理解图像，也不是负责融合条件，而是最后的“动作读出层”。

---

## 3. 在这份仓库里它长什么样

源码 `policy/ACT/detr/models/detr_vae.py`：

```python
self.action_head = nn.Linear(hidden_dim, state_dim)
```

前向中：

```python
a_hat = self.action_head(hs)
```

这里 `hs` 是 transformer decoder 输出。  
因此 action head 对应的就是：

```text
decoder hidden state -> linear -> predicted action
```

如果一次有 `num_queries` 个动作槽位，那么 action head 会对每个槽位都做同样映射，输出整个 action chunk。

---

## 4. 输入输出各是什么

### 输入：decoder 输出 `hs`

`hs` 表示每个 query 对应的最终 hidden representation。  
这些表示已经融合了：

- 当前视觉信息
- 当前机器人状态
- latent 条件
- chunk 内不同动作槽位之间的关系

所以 action head 输入的不是原始 token，而是“已经可以用来回归动作”的高层表示。

---

### 输出：`a_hat`

`a_hat` 就是预测的动作序列，形状上可以理解为：

```text
(bs, num_queries, state_dim)
```

含义是：

- batch 内每个样本
- 预测一个长度为 `num_queries` 的动作 chunk
- 每一步动作维度是 `state_dim`

在这份实现里，`state_dim` 就是动作维度，也和 `qpos` 维度一致地被使用。

---

## 5. 为什么 action head 只用一个线性层就够了

这是一个很值得理解的设计点。

很多人会直觉觉得：

- 动作输出这么重要
- 最后一层是不是应该再叠很多 MLP

但 ACT 这里没有这么做。原因是：

### 5.1 真正复杂的建模已经在 transformer 里完成了

到了 action head 之前，decoder hidden state 已经包含：

- 场景条件
- 机器人状态
- latent 高层模式
- 动作槽位关系

也就是说，action head 前的表示已经很“接近可输出动作”。

---

### 5.2 action head 更像线性读出器

它承担的是：

> 把已经准备好的动作语义表示，投影到真实控制空间。

所以它简单并不代表不重要，而是说明：

- 主要表示学习在前面完成
- 最后一层只做坐标系映射式读出

---

## 6. action head 和 query embeddings 的关系

这两者是前后相接的。

### query embeddings 决定“哪个槽位在预测哪一步动作”

见：[[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]

### action head 决定“这个槽位最后输出成什么动作向量”

也就是说：

- query 负责动作槽位身份
- decoder 负责把槽位表示建模好
- action head 负责把槽位表示变成动作数值

所以一条非常简洁的理解链是：

```text
query slot -> decoder hidden state -> action head -> action_t
```

---

## 7. action head 和 is_pad_head 的关系

这份实现里除了 action head，还有一个：

```python
self.is_pad_head = nn.Linear(hidden_dim, 1)
```

它们都读同一个 `hs`：

- `action_head(hs)`：输出动作预测 `a_hat`
- `is_pad_head(hs)`：输出 padding 相关预测 `is_pad_hat`

所以从结构上看，两个 head 是并行的。

但要特别注意：

> 在当前 `act_policy.py` 训练代码里，实际 loss 只显式使用了动作 L1 和 KL，并没有把 `is_pad_hat` 的监督项写进总损失。

因此当前仓库中，真正起主要训练作用的是 `action_head` 这一路。

---

## 8. action head 和训练损失怎么对应

训练时在 `ACTPolicy.__call__` 中：

```python
a_hat, is_pad_hat, (mu, logvar) = self.model(...)
all_l1 = F.l1_loss(actions, a_hat, reduction="none")
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
```

这说明：

- `action_head` 输出的 `a_hat` 会直接与真实动作 `actions` 做 L1 loss
- 对 padding 位置用 `~is_pad.unsqueeze(-1)` 做 mask
- 非 padding 区域的动作误差才真正参与训练

所以 action head 学到的就是：

> 在有效时间步上，把 decoder 表示尽量准确地回归到目标动作。

更完整见：[[RoboTwin/ACT/act_policy|act_policy]]

---

## 9. action head 在训练和推理时是否变化

### 训练时

- decoder hidden states -> action head -> `a_hat`
- `a_hat` 与真实动作做 L1

### 推理时

- decoder hidden states -> action head -> `a_hat`
- 不再计算 loss，直接把 chunk 作为候选动作序列输出

所以 action head 本身在训练和推理时没有结构变化。  
变化的是：

- latent 来源不同
- 推理阶段还会配合 chunk 选择或 temporal aggregation 使用输出动作

---

## 10. action head 输出后，动作是怎么被真正执行的

这里要和模型外层逻辑连起来看。

模型内部输出：

```text
a_hat: (bs, num_queries, state_dim)
```

但环境每一步只执行一个动作。  
所以外层 `ACT.get_action()` 会根据配置取出：

### 不开 temporal aggregation

直接取：

```python
raw_action = self.all_actions[:, self.t % self.query_frequency]
```

即从当前 chunk 中选择对应步的动作。

### 开 temporal aggregation

会把多个时刻预测出的重叠 chunk 存到 `all_time_actions`，然后对当前步所有可用动作做指数加权平均。

所以 action head 的输出是“动作 chunk 预测”，而不是“环境最终立刻执行的唯一动作”。

---

## 11. 看源码时怎么快速对应

你以后看到：

```python
self.action_head = nn.Linear(hidden_dim, state_dim)
a_hat = self.action_head(hs)
```

就要立刻反应：

- `hs` 是 decoder 的输出动作槽位表示
- action head 是最后一步线性读出
- 每个 query 一个动作位
- 整个 chunk 的动作是一起输出的

---

## 12. 一句话总结

> action head 是 ACT 中把 decoder 动作槽位表示映射成真实动作向量的最后读出层；
> 在 RoboTwin 实现里它就是一个 `Linear(hidden_dim, state_dim)`，主要表示学习发生在 transformer 中，而 action head 负责把最终表示转换成动作 chunk。
