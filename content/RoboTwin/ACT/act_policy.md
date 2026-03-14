---
title: act_policy
---

# act_policy

## 1. 这篇笔记要解决的问题

如果 `detr_vae.py` 解决的是“模型内部怎么前向”，
那么 `policy/ACT/act_policy.py` 解决的是：

1. 外层 policy 怎么调用模型
2. 训练时 loss 是怎么组织的
3. 推理时返回的动作 chunk 又是怎么被实际执行的

也就是说，这篇笔记讲的是“模型壳外面那一层”。

---

## 2. 文件里主要有哪几部分

这个文件里和 ACT 直接相关的主要有三块：

1. `ACTPolicy`
   - 训练 / 验证时调用模型并计算 loss
2. `kl_divergence`
   - 计算 KL 项
3. `ACT`
   - 部署时管理 chunk 输出、时序聚合、预处理和后处理

所以这个文件既不是纯模型定义文件，也不是纯训练脚本，
它更像一个：

> “模型调用 + loss 封装 + 推理执行逻辑”的中间层。

---

## 3. `ACTPolicy` 训练时到底做了什么

训练入口是：

```python
ACTPolicy.__call__(qpos, image, actions=None, is_pad=None)
```

当 `actions is not None` 时，走训练分支。

### 3.1 先归一化图像

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image = normalize(image)
```

也就是说，视觉输入不是直接原值送进模型，而是先按 ImageNet 风格均值方差归一化。

---

### 3.2 先裁动作长度

```python
actions = actions[:, :self.model.num_queries]
is_pad = is_pad[:, :self.model.num_queries]
```

这一步非常关键。

因为模型一次只输出 `num_queries` 个动作位置，而在本仓库里：

- `self.model.num_queries == chunk_size`

所以训练时并不是把整段原始动作序列全部拿来做监督，
而是只保留当前要预测的 chunk 长度。

这也说明：

> 外层 policy 会主动把训练目标对齐到模型当前 chunk 输出的长度。

---

### 3.3 调模型拿到三样结果

```python
a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
```

训练时模型返回：

- `a_hat`：预测动作 chunk
- `is_pad_hat`：padding 预测头输出
- `mu, logvar`：供 KL 使用的 posterior 参数

其中 `env_state = None`，所以这份实现实际主用的是：

- 图像
- qpos
- actions（训练时）

---

## 4. loss 是怎么组织的

### 4.1 KL 项

先调用：

```python
total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
```

`kl_divergence()` 的内部公式本质上就是标准高斯 KL：

\[
-\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)
\]

最后 `ACTPolicy` 实际使用的是：

```python
loss_dict["kl"] = total_kld[0]
```

也就是说，训练真正加到总损失里的，是 batch 上聚合后的总 KL。

### 4.2 动作 L1

```python
all_l1 = F.l1_loss(actions, a_hat, reduction="none")
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
```

这里的关键点有两个：

#### 第一，先逐元素算 L1

不是一上来直接平均，而是保留每个时间位、每个动作维度的误差。

#### 第二，用 `~is_pad` 做 mask

因为动作序列可能有 padding，所以这里会把 padding 位置屏蔽掉。

因此，L1 实际约束的是：

> 只有效动作位置的预测动作与真实动作要接近。

---

### 4.3 总损失

```python
loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
```

所以总损失就是：

\[
L = L_{L1} + \lambda_{KL} L_{KL}
\]

其中：

- `L1` 负责让输出动作贴近专家动作
- `KL` 负责约束 latent posterior 不要离标准高斯太远

这也是 ACT 训练时最核心的两个目标。

---

## 5. `L1` 和 `KL` 各自到底在约束什么

这个问题非常重要，因为如果只背公式，很容易不知道它们各自的功能分工。

### 5.1 L1 约束什么

L1 约束的是：

> 给定当前观测和 latent 条件后，decoder 输出的动作 chunk 必须接近专家动作 chunk。

它主要作用在：

- 主 transformer encoder
- decoder
- [[RoboTwin/ACT/concepts/action-head|action head]]
- 以及 latent 条件通路

因为整个网络最终都要对动作误差负责。

### 5.2 KL 约束什么

KL 约束的是：

> 训练时由 `qpos + actions` 推出来的 posterior 分布，不要偏离标准高斯太远。

它直接作用在：

- latent encoder
- `mu/logvar` 输出
- latent 空间分布结构

它的意义不是让动作更准，而是让 latent 空间更规整，
从而保证推理时没有真实动作也能直接用 `z=0`。

所以一句话记：

- `L1` 解决“动作要预测对”
- `KL` 解决“latent 空间要可用、可泛化”

---

## 6. 一个很容易忽略的实现细节：`is_pad_head` 没被单独监督

模型 forward 会返回：

- `is_pad_hat`

但在当前 `ACTPolicy.__call__` 的训练分支里，没有看到类似：

- BCE
- CE
- 或别的 `is_pad_hat` loss

也就是说：

> 当前这份 RoboTwin 实现里，`is_pad_head` 被定义出来了，但训练时并没有单独把它放进总损失。

这个细节很值得记，因为你之后看模型结构时很容易下意识以为它一定被训练了。

---

## 7. 推理时 `ACTPolicy` 又做了什么

当 `actions is None` 时，`ACTPolicy` 走推理分支：

```python
a_hat, _, (_, _) = self.model(qpos, image, env_state)
return a_hat
```

所以它在推理时只返回：

- 预测动作 chunk `a_hat`

这里不会计算：

- L1
- KL
- 任何 loss

所以 `ACTPolicy` 可以理解为：

- 训练时：返回 `loss_dict`
- 推理时：返回动作 chunk

---

## 8. `ACT` 类在部署时又做了什么

`ACT` 类包在 `ACTPolicy` 外面，主要负责部署时真正和环境交互的逻辑。

它做的事有四类。

### 8.1 加载 checkpoint 和数据统计量

它会尝试从 `ckpt_dir` 下读取：

- `dataset_stats.pkl`
- `policy_last.ckpt`

这样部署时就可以做：

- qpos 归一化
- action 反归一化
- 模型权重加载

### 8.2 预处理 qpos

```python
(qpos - qpos_mean) / qpos_std
```

### 8.3 后处理 action

```python
action * action_std + action_mean
```

### 8.4 组织 chunk 动作的实际执行

也就是：

- 何时重新 query 模型
- 当前时刻从 chunk 中取哪一个动作
- 是否做 temporal aggregation

---

## 9. 不开 temporal aggregation 时，动作怎么执行

如果 `temporal_agg=False`，那么：

- `query_frequency = num_queries`
- 不是每一步都重新跑模型
- 而是每隔一个 chunk 长度才重新预测一次

然后当前时刻取：

```python
raw_action = self.all_actions[:, self.t % self.query_frequency]
```

也就是说：

> 一次前向拿到一个动作 chunk，随后连续若干环境步依次取出其中对应位置的动作。

这正是 ACT chunked action 的落地方式。

---

## 10. 开 temporal aggregation 时，动作怎么执行

如果 `temporal_agg=True`，那么：

- `query_frequency = 1`
- 每一步都重新预测一个新的 chunk
- 所有历史 chunk 对“当前时刻”的建议动作都会被收集起来
- 再用指数权重做加权平均

代码里对应：

```python
self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
actions_for_curr_step = self.all_time_actions[:, self.t]
...
raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
```

所以 temporal aggregation 的本质是：

> 对当前时刻，从多个 chunk 给出的候选动作中做时间加权融合，而不是只相信最近一次预测。

---

## 11. 这份文件在整套 ACT 结构中的位置

你可以把整个 RoboTwin ACT 代码理解成三层：

### 第一层：模型内部

- [[RoboTwin/ACT/detr_vae|detr_vae]]
- [[RoboTwin/ACT/transformer-dataflow|transformer-dataflow]]

负责：模型怎么前向。

### 第二层：策略封装

- `act_policy.py`

负责：怎么喂输入、怎么裁动作、怎么算 loss、怎么把 chunk 真正拿去执行。

### 第三层：训练脚本 / 数据脚本

例如：

- `imitate_episodes.py`

负责：训练循环、数据集、日志、保存等。

所以 `act_policy.py` 的位置非常适合当作：

> “模型和训练/部署流程之间的桥梁层”。

---

## 12. 一句话总结

> `act_policy.py` 不负责定义 ACT 主模型结构，而是负责把 `DETRVAE` 包装成一个真正可训练、可推理、可部署的策略：训练时组织 `L1 + KL`，推理时返回 chunk 动作，部署时再决定如何从 chunk 中取当前动作以及是否做 temporal aggregation。
