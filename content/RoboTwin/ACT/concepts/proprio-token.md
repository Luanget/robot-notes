---
title: proprio token
---

# proprio token

## 1. 什么是 proprio token

在 ACT 里，`proprio token` 表示机器人自身状态对应的 token。  
在你这份 RoboTwin 仓库实现里，它具体对应的是：

> 当前时刻的 `qpos` 经过一个线性层投影到 `hidden_dim` 之后得到的状态 token。

源码位置在 `policy/ACT/detr/models/detr_vae.py`：

```python
self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
```

前向里：

```python
proprio_input = self.input_proj_robot_state(qpos)
```

因此，proprio token 不是从图像里来的，也不是从动作序列里来的，而是：

- 直接由机器人关节状态 `qpos`
- 投影到 transformer 的隐藏维度
- 作为 encoder 侧的一个额外条件 token

---

## 2. 为什么 ACT 需要 proprio token

如果只有图像，没有机器人自身状态，那么模型只能看到“外部世界长什么样”，但不知道：

- 当前手臂已经抬到哪里了
- 夹爪现在是张开还是闭合
- 左右臂当前关节位置是什么

而动作生成本质上必须依赖当前状态。  
同样看到一张图：

- 如果机械臂已经靠近物体，下一步可能是抓取
- 如果机械臂还很远，下一步可能是先接近

这些差别，单靠图像未必足够稳定地区分，特别是多视角图像中机器人自身也会被部分遮挡、投影、失真。

所以 proprio token 的作用就是显式告诉模型：

> **当前机器人自己处于什么状态。**

---

## 3. 在这份仓库里，proprio token 来自什么

这里要特别注意：

### 3.1 它来自 `qpos`

也就是 observation 里的机器人关节状态。

外层调用链上：

- 训练时，数据集会提供 `qpos`
- 推理时，`ACT.get_action()` 从 `obs["qpos"]` 取出状态
- 然后做标准化 `pre_process`
- 再送入 policy

最终传到模型里的是标准化后的 `qpos`。

---

### 3.2 它不是时间序列 token

在当前这份 ACT 实现里，主 transformer 输入的 proprio 只有“当前时刻的一个状态 token”，不是过去很多时刻的状态序列。

也就是说，这里并没有显式把：

```text
qpos_{t-3}, qpos_{t-2}, qpos_{t-1}, qpos_t
```

一起送进主 transformer。  
主 transformer 侧看到的是“当前时刻的 robot state 条件”。

---

## 4. 代码里它是怎么被构造的

在 `DETRVAE.forward(...)` 中：

```python
proprio_input = self.input_proj_robot_state(qpos)
```

其中：

- 输入：`qpos`，形状大致是 `(bs, state_dim)`
- 输出：`proprio_input`，形状大致是 `(bs, hidden_dim)`

然后它不会像图像那样先变成 feature map，而是直接作为一个额外 token 被送进主 transformer。

因此 proprio token 的结构很简单：

```text
qpos  --Linear(state_dim -> hidden_dim)-->  proprio token
```

---

## 5. proprio token 在 encoder 序列里的位置

在这份实现里，主 transformer encoder 接收两类输入：

1. 图像 flatten 后的一串 image tokens
2. 额外 prepend 的 tokens：latent token 与 proprio token

因此从语义上看，encoder 输入顺序可以理解为：

```text
[latent token, proprio token, image tokens...]
```

这里 proprio token 不是混在图像 patch 序列中间，而是和 latent token 一样，作为“额外条件 token”加到序列前部。

它对应的位置编码来自：

```python
self.additional_pos_embed = nn.Embedding(2, hidden_dim)
```

其中两个 learned positional embeddings 分别服务于：

- latent token
- proprio token

所以 proprio token 除了有自身内容表示，还带有“我是 proprio 这一类 token”的位置信息。

---

## 6. proprio token 和 latent token 的区别

这两个都不是图像 patch token，所以很容易混。

### proprio token 表达的是：当前真实状态

它来自 `qpos`，是确定的、观测到的机器人状态。

---

### latent token 表达的是：动作模式条件

它来自 CVAE latent：

- 训练时由 posterior encoder 从 `qpos + actions` 推出
- 推理时直接用零向量代替

它更多承担的是“高层动作风格 / 计划摘要”的角色。

所以这两个 token 的语义完全不同：

- proprio：现在机器人在哪里、姿态是什么
- latent：这一段动作应该遵循什么潜在模式

见：[[RoboTwin/ACT/concepts/latent-token|latent token]]

---

## 7. proprio token 和 image tokens 的区别

### image tokens 提供外部视觉环境

它们告诉模型：

- 目标物体在哪里
- 场景布局如何
- 机器人在图像中的相对位置如何

### proprio token 提供内部状态

它告诉模型：

- 关节角 / 机械臂配置
- 夹爪状态
- 当前自身姿态

所以 ACT 的动作生成不是“只看图像”，而是依赖：

- 外部世界：image tokens
- 自身状态：proprio token
- 高层动作模式：latent token

这三类条件共同进入 encoder，形成 [[RoboTwin/ACT/concepts/memory|memory]]。

---

## 8. 为什么不能只靠图像替代 proprio token

理论上，图像里也能看到机械臂，因此有人会问：

> “既然相机能拍到机器人，为什么还要再输入 qpos？”

原因主要有三点。

### 8.1 图像对机器人状态的表达不精确

图像中的机械臂位置会受到：

- 视角变化
- 遮挡
- 透视失真
- 分辨率限制

影响。  
而 `qpos` 是直接、精确的状态量。

---

### 8.2 控制任务对状态精度要求很高

模仿学习策略最终输出的是动作，而动作必须和当前状态严密对应。  
很多时候图像上看起来差不多，但关节实际已经有明显差异，这会影响下一步控制。

---

### 8.3 跨视角整合成本高

如果完全依赖图像，模型需要从多个相机视角里反推出当前 robot configuration。  
而 proprio token 直接把这部分信息显式提供出来，能显著减轻建模难度。

---

## 9. 在训练和推理中的作用是否变化

proprio token 在训练与推理中的路径其实基本不变。

### 训练时

- `qpos` 投影成 proprio token
- 和 latent token、image tokens 一起送入 encoder
- decoder 再基于 memory 预测动作 chunk

### 推理时

- `qpos` 依然投影成 proprio token
- 唯一主要变化是 latent token 的来源不同：不再来自 posterior，而是来自零向量

所以可以说：

> proprio token 是 ACT 在训练和推理中都稳定存在的一条条件通路。

---

## 10. 与外层归一化的关系

这一点很贴 RoboTwin 实现，也很值得记。

`ACT.get_action()` 在推理时会先对 `qpos` 做标准化：

```python
qpos_normalized = self.pre_process(qpos_numpy)
```

如果加载了 `dataset_stats.pkl`，会执行：

```python
(qpos - qpos_mean) / qpos_std
```

训练数据通常也是按同样统计量归一化后喂给模型。  
所以 proprio token 实际编码的，不是未经处理的原始 qpos，而是：

> 经过数据集统计量标准化后的 robot state。

这也是为什么 checkpoint 推理时最好同时带上 `dataset_stats.pkl`。

---

## 11. 它最终如何影响动作输出

proprio token 自己不会直接变成动作。  
它的作用路径是：

1. `qpos` -> proprio token
2. proprio token 与 latent / image 一起进入 encoder
3. encoder 输出 [[RoboTwin/ACT/concepts/memory|memory]]
4. decoder queries 从 memory 中读取和当前动作槽位相关的信息
5. 最终经 [[RoboTwin/ACT/concepts/action-head|action head]] 输出动作 chunk

因此，proprio token 对动作的影响是“条件性影响”，不是直接线性映射。

---

## 12. 看源码时怎么一眼对上

你以后看到这几行，就要立刻反应过来：

```python
self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
proprio_input = self.input_proj_robot_state(qpos)
```

它们对应的就是：

- `qpos` 的 token 化
- 把机器人当前状态接入主 transformer
- 作为 encoder memory 的一部分参与后续动作生成

---

## 13. 一句话总结

> proprio token 是当前机器人关节状态 `qpos` 经过线性投影后得到的状态 token；
> 它和 latent token、image tokens 一起进入主 transformer encoder，提供动作生成所必需的“当前自身状态”条件。
