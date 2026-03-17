---
title: ACT 代码文件地图
---

# ACT 代码文件地图

## 1. 这篇笔记要解决的问题

这篇不是讲某个单独概念，而是帮助你在不反复翻源码的情况下，快速回答下面这些问题：

1. RoboTwin 的 ACT 实现主要分布在哪些文件里
2. 每个文件各自负责什么
3. 如果我想看某个概念，第一站应该去哪个文件
4. 训练、推理、loss、transformer、latent、backbone 分别在哪一层实现

这篇可以当作你整套 ACT 笔记的“源码导航页”。

推荐和下面这些页面配合看：

- [[RoboTwin/ACT/index|ACT 学习入口]]
- [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]
- [[RoboTwin/ACT/train-and-infer|训练与推理流程]]
- [[RoboTwin/ACT/detr_vae|detr_vae]]
- [[RoboTwin/ACT/act_policy|act_policy]]
- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流转]]
- [[RoboTwin/ACT/dataset-and-dataloader|Dataset 与 Dataloader 数据流]]

---

## 2. ACT 在 RoboTwin 仓库里的主文件分布

你现在最需要记住的核心文件，就是下面这 6 个：

```text
policy/ACT/utils.py
policy/ACT/act_policy.py
policy/ACT/imitate_episodes.py
policy/ACT/detr/models/detr_vae.py
policy/ACT/detr/models/transformer.py
policy/ACT/detr/models/backbone.py
```

如果只从“快速建立文件职责感”出发，可以先这样粗记：

- `utils.py`：dataset、dataloader、归一化统计量、样本构造
- `act_policy.py`：训练包装层 + 推理封装层
- `imitate_episodes.py`：训练 / 验证 / rollout 脚本逻辑
- `detr_vae.py`：ACT 模型主体，负责 latent、视觉、transformer、动作头的总装
- `transformer.py`：主 transformer 细节实现
- `backbone.py`：视觉 backbone，把图像变成 feature map 和位置编码

---

## 3. 每个核心文件到底负责什么


## 3.0 `policy/ACT/utils.py`

这一块是你理解 dataset / dataloader 时的第一站。

### 它主要负责：

1. 扫描整个数据集，计算 `qpos / action` 的统计量
2. 求全局 `max_action_len`
3. 定义 `EpisodicDataset`，构造单样本
4. 定义 `load_data(...)`，构造 train / val dataloader

### 你最应该关注的对象

#### `get_norm_stats(...)`

它会遍历所有 `episode_i.hdf5`，读出整段 `qpos` 和 `action`，先 pad 到统一长度，再计算：

- `action_mean / action_std`
- `qpos_mean / qpos_std`
- `max_action_len`

这一步不是边角料，而是后面 `EpisodicDataset` 能正确归一化样本、以及 DataLoader 能默认 stack 的前提。

#### `EpisodicDataset`

它不是纯图像 dataset，而是联合构造：

- `image_data`
- `qpos_data`
- `action_data`
- `is_pad`

其中图像会从 HDF5 中读取单时刻多相机观测，组装成 `[num_cams, C, H, W]`。

#### `load_data(...)`

它负责：

- train / val 按 episode 划分
- 构造 `train_dataset / val_dataset`
- 构造 `train_dataloader / val_dataloader`

这部分最好和 [[RoboTwin/ACT/dataset-and-dataloader|Dataset 与 Dataloader 数据流]] 一起看。

---

## 3.1 `policy/ACT/act_policy.py`

这是你理解“外层 policy 如何调用模型”的第一站。

### 它主要负责：

1. 构建 policy 对象
2. 包装 `build_ACT_model_and_optimizer(...)`
3. 训练时调用模型并组织 loss
4. 推理时调用模型输出动作 chunk
5. 部署时做输入归一化、输出反归一化、temporal aggregation

### 你最应该关注的类

#### `ACTPolicy`

这是训练 / 验证阶段最核心的 policy 封装。

它的关键逻辑是：

- 先对图像做 ImageNet 风格标准化
- 训练时裁切 `actions[:, :num_queries]`
- 调 `self.model(...)` 得到 `a_hat, is_pad_hat, (mu, logvar)`
- 算 `l1 + kl * kl_weight`
- 推理时不传入 `actions`，直接让模型输出 `a_hat`

这部分最适合配合 [[RoboTwin/ACT/act_policy|act_policy]] 一起看。

---

#### `ACT`

这是更偏部署 / rollout 侧的封装。

它的职责包括：

- 加载 checkpoint
- 加载 `dataset_stats.pkl`
- `pre_process` 做 `qpos` 归一化
- `post_process` 做动作反归一化
- `get_action` 中按照固定频率查询 policy
- 可选 temporal aggregation，把多个 chunk 对当前时刻的动作预测做加权融合

所以如果你想知道：

- 实际跑起来时模型多久 query 一次
- 输出 chunk 后怎么取当前动作
- temporal aggregation 怎么接进去

第一站应该看这里。

---

## 3.2 `policy/ACT/imitate_episodes.py`

这是训练流程层面的关键文件。

虽然你在理解模型结构时不一定先读它，但如果你要把“模型如何被训练脚本驱动起来”串通，它非常重要。

### 它主要负责：

1. 读取训练参数
2. 构建 dataloader
3. 构建 policy
4. 训练 / 验证循环
5. 保存 checkpoint 和统计信息
6. rollout / evaluation 阶段的动作执行逻辑

### 它的价值是什么

`detr_vae.py` 和 `transformer.py` 告诉你“模型内部怎么流”；
而 `imitate_episodes.py` 告诉你：

> 这套模型在训练脚本里是怎样被真正跑起来的。

所以如果你要理解：

- chunk size 在训练脚本里怎么被设置
- temporal aggregation 在脚本里如何影响 query frequency
- 评估时动作是怎样从 chunk 里拿出来的

这个文件非常关键。

---

## 3.3 `policy/ACT/detr/models/detr_vae.py`

这是整套 ACT 模型的主体装配文件，也是最值得反复看的一个文件。

### 它主要负责：

1. 定义 `DETRVAE` 类
2. 定义 `action_head`、`is_pad_head`、`query_embed`
3. 定义 posterior encoder 所需的 latent 模块
4. 定义视觉 backbone 接口后的输入投影
5. 在 `forward()` 中组织训练 / 推理两条数据流
6. 调用主 transformer
7. 输出动作 chunk 和 latent 统计量

### 你可以把它理解成什么

它相当于一个“总装厂”：

- latent 分支在这里接入
- 图像 backbone 在这里接入
- qpos 投影在这里接入
- transformer 在这里接入
- action head 在这里接入

所以如果你脑子里只允许留一个“模型总图文件”，那就是它。

这部分最好结合 [[RoboTwin/ACT/detr_vae|detr_vae]] 看。

---

### `detr_vae.py` 里最关键的几个对象

#### `self.query_embed`

```python
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

对应动作槽位的 learned queries。

详见 [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]。

---

#### `self.action_head`

```python
self.action_head = nn.Linear(hidden_dim, state_dim)
```

把 decoder 输出的 hidden states 直接映射到动作维度。

详见 [[RoboTwin/ACT/concepts/action-head|action head]]。

---

#### `self.is_pad_head`

```python
self.is_pad_head = nn.Linear(hidden_dim, 1)
```

结构上用来预测 padding，但当前 `act_policy.py` 里没有把它显式纳入 loss。

这一点很容易在读代码时忽略。

---

#### latent 相关模块

包括：

- `cls_embed`
- `encoder_action_proj`
- `encoder_joint_proj`
- `latent_proj`
- `latent_out_proj`

这些模块共同组成训练时的 posterior encoder 路径。

详见 [[RoboTwin/ACT/concepts/latent-token|latent token]]。

---

## 3.4 `policy/ACT/detr/models/transformer.py`

如果说 `detr_vae.py` 告诉你“零件有哪些”，那 `transformer.py` 告诉你“零件之间怎么流”。

### 它主要负责：

1. encoder / decoder 结构定义
2. flatten 图像 feature map 成 token 序列
3. 把 `latent_input` 和 `proprio_input` prepend 到 encoder 输入前面
4. 构造零初始化 `tgt`
5. 用 `query_embed` 作为 decoder 查询位
6. 逐层执行 self-attention / cross-attention / FFN

### 你最该在这里找什么

如果你想确认下面这些问题，第一站就该是这个文件：

- image tokens 在哪一步形成
- memory 到底是什么形状 / 什么含义
- 为什么 `tgt` 可以是全零
- query 和 memory 是怎样做 cross-attention 的
- decoder 输出 `hs` 到底是什么

这部分最好结合 [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流转]] 一起看。

---

## 3.5 `policy/ACT/detr/models/backbone.py`

这是视觉输入的入口。

### 它主要负责：

1. 构建 backbone 网络
2. 从输入图像提取 feature map
3. 生成对应的位置编码 `pos`
4. 把图像表示交给上层 `detr_vae.py`

### 它在整条主线中的位置

可以把它理解成：

```text
image -> backbone.py -> feature map + pos -> detr_vae.py -> transformer.py
```

也就是说，它本身不负责动作预测，而是负责把图像变成后续 transformer 可读的视觉表示。

如果你要看“图像是怎么先变成 feature 的”，就应该来这里。

---

## 4. 按问题来反推：该先看哪个文件

这一节很适合做复习时的“问题索引”。

### 4.1 我想看训练 loss 怎么算

先看：

- `policy/ACT/act_policy.py`

重点关注：

- `ACTPolicy.__call__`
- `F.l1_loss(..., reduction="none")`
- `kl_divergence(mu, logvar)`
- `loss_dict["loss"] = l1 + kl * kl_weight`

配套笔记：[[RoboTwin/ACT/act_policy|act_policy]]

---

### 4.2 我想看训练和推理为什么不同

先看：

- `policy/ACT/detr/models/detr_vae.py`
- `policy/ACT/act_policy.py`

重点关注：

- `is_training = actions is not None`
- 训练时 posterior encoder 分支
- 推理时 `latent_sample = zeros(...)`

配套笔记：[[RoboTwin/ACT/train-and-infer|训练与推理流程]]

---

### 4.3 我想看 latent token 从哪来

先看：

- `policy/ACT/detr/models/detr_vae.py`

重点关注：

- `encoder_action_proj`
- `encoder_joint_proj`
- `cls_embed`
- `latent_proj`
- `latent_out_proj`

配套笔记：[[RoboTwin/ACT/concepts/latent-token|latent token]]

---

### 4.4 我想看 query 为什么对应动作槽位

先看：

- `policy/ACT/detr/models/detr_vae.py`
- `policy/ACT/detr/models/transformer.py`

重点关注：

- `self.query_embed = nn.Embedding(num_queries, hidden_dim)`
- `tgt = torch.zeros_like(query_embed)`
- decoder self-attention / cross-attention
- `a_hat = self.action_head(hs)`

配套笔记：[[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]

---

### 4.5 我想看图像怎么进入 transformer

先看：

- `policy/ACT/detr/models/backbone.py`
- `policy/ACT/detr/models/detr_vae.py`
- `policy/ACT/detr/models/transformer.py`

重点关注：

- backbone 输出 `features, pos`
- `input_proj`
- 多相机 feature 拼接
- `flatten(2).permute(2,0,1)`

配套笔记：[[RoboTwin/ACT/concepts/image-tokens|image tokens]]

---

### 4.6 我想看 rollout 时当前动作怎么从 chunk 里取出来

先看：

- `policy/ACT/act_policy.py` 中的 `ACT.get_action`
- `policy/ACT/imitate_episodes.py`

重点关注：

- `self.query_frequency`
- `self.all_actions`
- `raw_action = self.all_actions[:, self.t % self.query_frequency]`
- temporal aggregation 相关逻辑

配套笔记：[[RoboTwin/ACT/train-and-infer|训练与推理流程]]

---

## 5. 一条最适合你的阅读顺序

如果你的目标是：

> 不陷进源码细枝末节，但能把 ACT 在 RoboTwin 里的实现迅速串起来

我建议按下面顺序看。

### 第一步：总览

先看：

- [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]
- [[RoboTwin/ACT/train-and-infer|训练与推理流程]]

目标：

- 先把训练 / 推理主线建立起来
- 知道 posterior 只在训练时出现
- 知道推理时是整 chunk 输出，不是一步一步生成

---

### 第二步：模型总装图

再看：

- [[RoboTwin/ACT/detr_vae|detr_vae]]

目标：

- 认识 `query_embed`、`action_head`、latent 模块、visual 模块在总图里的位置

---

### 第三步：主 transformer 内部数据流

再看：

- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流转]]

目标：

- 知道 memory 是怎么来的
- 知道 query 是怎么读取 memory 的

---

### 第四步：概念页补齐

最后查概念页：

- [[RoboTwin/ACT/concepts/latent-token|latent token]]
- [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]
- [[RoboTwin/ACT/concepts/image-tokens|image tokens]]
- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- [[RoboTwin/ACT/concepts/memory|memory]]
- [[RoboTwin/ACT/concepts/action-head|action head]]

目标：

- 把容易混淆的局部概念补扎实

---

### 第五步：回源码时按问题找文件

最后再用这篇“代码文件地图”反查源码。

这样你不是漫无目的地翻文件，而是：

- 先有知识结构
- 再带着问题回源码找定位

效率会高很多。

---

## 6. 最容易搞混的几组文件关系

### 6.1 `act_policy.py` 和 `detr_vae.py`

- `act_policy.py`：外层训练 / 推理封装，负责 loss 和调用方式
- `detr_vae.py`：模型主体，负责内部前向结构

可以理解成：

> `act_policy.py` 是“怎么用模型”，`detr_vae.py` 是“模型内部怎么长”。

---

### 6.2 `detr_vae.py` 和 `transformer.py`

- `detr_vae.py`：负责把各个输入零件装配好，并调用 transformer
- `transformer.py`：负责 encoder / decoder 的内部注意力计算和 token 流动

可以理解成：

> `detr_vae.py` 管总装，`transformer.py` 管核心流转。

---

### 6.3 `backbone.py` 和 `transformer.py`

- `backbone.py`：把图像变成 feature map
- `transformer.py`：把这些 feature map 当成 token 序列进一步融合和解码

可以理解成：

> `backbone.py` 负责“看见”，`transformer.py` 负责“理解并生成动作”。

---

## 7. 一句话总结

我认为 RoboTwin 里的 ACT 代码文件地图，可以这样记：

> `backbone.py` 负责把图像变成视觉特征，`transformer.py` 负责让 token 真正流起来，`detr_vae.py` 负责把 latent / qpos / vision / query 全部装成一个 ACT 模型，`act_policy.py` 负责训练损失与推理封装，`imitate_episodes.py` 负责把整套东西在训练和 rollout 脚本里真正跑起来。
