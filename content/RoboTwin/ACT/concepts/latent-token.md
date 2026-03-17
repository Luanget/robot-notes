---
title: latent token
---

# latent token

## 1. latent token 是什么

在 RoboTwin 这份 ACT 实现里，latent token 是：

> 由潜变量 `z` 投影得到的一个条件 token，用来告诉主 transformer“这次动作应当采用哪一种轨迹模式”。

它不是图像 token，也不是机器人状态 token，更不是直接动作输出。

它的位置是在主 encoder 输入序列最前面，和 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]、[[RoboTwin/ACT/concepts/image-tokens|image tokens]] 一起参与融合。

---

## 2. 在这份实现里它从哪里来

### 训练时

训练时 latent token 来自 posterior encoder：

```text
qpos + actions -> latent encoder -> mu, logvar -> sample z -> latent_out_proj -> latent token
```

具体文件在：

- `policy/ACT/detr/models/detr_vae.py`

这里有两个非常值得记住的实现细节：

1. posterior encoder 的输入只有 `qpos + actions`，**不含图像**
2. 这份实现里 `latent_dim = 32`

也就是说，训练时 latent token 代表的是：

> 当前状态下，这条真实专家动作序列对应的动作模式摘要。

### 推理时

推理时没有真实未来动作，因此不再构造 posterior。
代码直接使用：

```python
latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
latent_input = self.latent_out_proj(latent_sample)
```

所以推理时不是随机采样，也不是从别的网络预测 prior，而是直接：

> 令 `z = 0`，再投影成 latent token。

---

## 3. 为什么训练时要有 latent token

因为 ACT 要解决的不是“当前观测只对应唯一一个未来动作序列”。

在很多机器人任务里，同样的观测下，存在多种都能成功的动作模式，例如：

- 从左侧靠近
- 从右侧靠近
- 抓取前先略微调整姿态

如果模型只学一个确定性映射，很容易把这些模式平均掉，得到一个折中但不可靠的动作。

所以 latent token 的作用是：

> 给主 transformer 一个额外条件，告诉它“这次该走哪一类动作模式”。

---

## 4. 它在主 transformer 中是怎么用的

在 `transformer.py` 中，图像特征 flatten 后，会把：

- `latent_input`
- `proprio_input`

stack 成两个额外 token：

```python
addition_input = torch.stack([latent_input, proprio_input], axis=0)
src = torch.cat([addition_input, src], axis=0)
```

因此主 encoder 实际看到的序列是：

```text
[latent, proprio, image tokens...]
```

这说明 latent token 的作用不是最后再拼接给动作头，而是：

> 从 encoder 融合阶段开始，就参与整段条件序列的 self-attention。

所以：

- 图像 token 可以读取 latent 条件
- proprio token 也可以读取 latent 条件
- 最终 encoder 输出的 [[RoboTwin/ACT/concepts/memory|memory]] 会带着 latent 影响

---

## 5. 为什么推理时 `z=0` 还能工作

因为训练时外层 policy 会加入 KL：

\[
D_{KL}(q(z|qpos, actions) \parallel \mathcal{N}(0, I))
\]

这会强迫 posterior 不要离标准高斯太远。

于是推理时虽然没有真实动作去构造 posterior，
但直接使用 prior 的中心点 `z=0`，通常仍能落在合理 latent 区域中。

所以应该这样记：

> 训练时通过 KL 把 latent 空间约束成“以 0 为中心、结构较规整”的空间，因此推理时直接用 `z=0` 才可行。

---

## 6. 最容易混淆的几个点

### 6.1 latent token 不是图像提出来的

图像来自 backbone；latent token 来自 posterior 分支。
这两条路径是分开的。

### 6.2 latent token 不是 decoder query

latent token 属于 **encoder 输入条件**；
[[RoboTwin/ACT/concepts/query-embeddings|query embeddings]] 属于 **decoder 输出槽位**。

### 6.3 latent token 不是“动作本身”

它只是动作模式条件，不直接等于某一步动作值。

---

## 7. 复习时最应该立刻反应出来的点

1. 这份实现里 `latent_dim = 32`
2. posterior encoder 输入是 `qpos + actions`，不看图像
3. 训练时 `mu/logvar -> sample z -> latent_out_proj -> latent token`
4. 推理时直接 `z=0`
5. latent token 会和 proprio、image tokens 一起进入主 encoder

---

## 8. 一句话总结

> 在 RoboTwin 的 ACT 里，latent token 是由潜变量 `z` 投影得到的条件 token：训练时它来自 `qpos + actions` 构造的 posterior，推理时则直接由 `z=0` 得到，用来告诉主 transformer这次动作应当遵循哪种轨迹模式。
