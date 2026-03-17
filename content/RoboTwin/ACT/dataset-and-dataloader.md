---
title: Dataset 与 Dataloader 数据流
---

# Dataset 与 Dataloader 数据流

## 1. 这篇笔记要解决的问题

前面几篇主线笔记已经把 ACT 的模型主体讲得比较完整了：

- [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]] 讲整体链路
- [[RoboTwin/ACT/train-and-infer|训练与推理流程]] 讲训练 / 推理分叉
- [[RoboTwin/ACT/detr_vae|DETR-VAE 主体结构]] 讲模型总装
- [[RoboTwin/ACT/act_policy|ACT Policy 与损失组织]] 讲外层调用和 loss
- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流]] 讲主 transformer 内部

但是如果你要真正把“图像、qpos、actions 是怎么被整理成一个 batch，并送进 ACT”的过程讲清楚，还缺一块专门的笔记。

这篇就专门解决下面这些问题：

1. 在 RoboTwin 这份实现里，dataset / dataloader 主要分布在哪些文件
2. `utils.py` 里的 `EpisodicDataset` 到底怎么构造单个样本
3. 图像在 dataset 中是怎么从 HDF5 读出来、堆成多相机张量的
4. `get_norm_stats(...)` 为什么是 dataloader 构造前的必要步骤
5. DataLoader 为什么可以用默认 collate 直接拼 batch
6. batch 进入 `imitate_episodes.py` 和 `act_policy.py` 之后又发生了什么

这篇的目标不是替代源码，而是尽量做到：

> 不打开源码时，也能迅速回忆“这份 RoboTwin 里的 ACT 样本是怎么构造出来的”。

---

## 2. 这一块代码到底分布在哪

如果只问“dataset / dataloader 主体在哪”，答案确实主要是：

```text
policy/ACT/utils.py
```

但如果你要把链条讲完整，就不能只停在 `utils.py`。因为真正完整的数据流是：

```text
policy/ACT/utils.py
-> policy/ACT/imitate_episodes.py
-> policy/ACT/act_policy.py
```

三者分工如下：

### `policy/ACT/utils.py`

这一块是主体，负责：

- `get_norm_stats(...)`：扫描整个数据集，算 qpos / action 统计量和 `max_action_len`
- `EpisodicDataset`：构造单样本
- `load_data(...)`：构造 train / val dataloader

### `policy/ACT/imitate_episodes.py`

这一块负责：

- 从 task config 里拿到 `dataset_dir / num_episodes / camera_names`
- 调用 `load_data(...)`
- 在训练循环里接住 dataloader 返回的 batch
- 把 batch 搬到 GPU 后传给 policy

### `policy/ACT/act_policy.py`

这一块负责：

- 对 dataloader 输出的图像 batch 再做 resize
- 再做 ImageNet mean/std 标准化
- 然后再送进模型主体

所以一句话记：

> dataset / dataloader 的主体在 `utils.py`，但它的后继约束在 `imitate_episodes.py` 和 `act_policy.py` 里。

---

## 3. 从文件角度看，这条数据流的完整链路是什么

你可以先把整条链路记成：

```text
processed_data/episode_i.hdf5
-> get_norm_stats(...) 先扫全数据集，算统计量和最大长度
-> EpisodicDataset.__getitem__() 构造单样本
-> DataLoader 默认 collate 拼 batch
-> imitate_episodes.forward_pass(...) 接 batch
-> ACTPolicy.__call__(...) 进一步 resize + normalize 图像
-> DETRVAE forward
```

其中，一个单样本不是“单张图像”这种简单结构，而是四元组：

```python
(image_data, qpos_data, action_data, is_pad)
```

所以这份 dataset 不是图像分类式 dataset，而是：

> 当前观测（图像 + qpos） + 未来动作监督（action + is_pad） 的联合样本构造器。

---

## 4. 先看 dataloader 构造前的准备：`get_norm_stats(...)`

位置：

```text
policy/ACT/utils.py
```

函数原型：

```python
def get_norm_stats(dataset_dir, num_episodes):
```

很多人第一次看会把它当作“辅助函数”，但在这份实现里，它其实是 dataloader 构造前的必要准备。

因为后面 `EpisodicDataset` 要依赖它给出的两类东西：

1. `norm_stats`
2. `max_action_len`

---

### 4.1 它先遍历整个数据集

函数一开始会遍历所有 `episode_i.hdf5`：

```python
all_qpos_data = []
all_action_data = []
for episode_idx in range(num_episodes):
    dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
    all_qpos_data.append(torch.from_numpy(qpos))
    all_action_data.append(torch.from_numpy(action))
```

这说明：

- 统计量是在**整个数据集范围**上算的
- 不是 per-episode
- 也不是在训练中动态估计

而且这里读的是整段 qpos / action 序列，而不是只取某个 timestep。

---

### 4.2 它先求出最大长度

```python
max_qpos_len = max(q.size(0) for q in all_qpos_data)
max_action_len = max(a.size(0) for a in all_action_data)
```

这里最重要的是：

- `max_action_len` 会被传给 `EpisodicDataset`
- 之后 `__getitem__` 里会把动作后缀 pad 到这个长度

所以 DataLoader 能顺利拼 batch，前提之一就是这里先把统一长度标准定出来。

---

### 4.3 它会先对 qpos / action 做 pad，再算统计量

这一步很容易被忽略。

对 qpos：

```python
if current_len < max_qpos_len:
    pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
    qpos = torch.cat([qpos, pad], dim=0)
```

对 action：

```python
if current_len < max_action_len:
    pad = action[-1:].repeat(max_action_len - current_len, 1)
    action = torch.cat([action, pad], dim=0)
```

注意这里不是补零，而是**重复最后一个元素**。

也就是说，这一步 pad 的目的不是给训练样本用，而是为了让所有 episode 的时间长度先对齐，再统一求均值 / 方差。

这一点要和 `__getitem__` 中训练样本的 `padded_action = np.zeros(...)` 分开理解：

- `get_norm_stats(...)` 里的 pad：为了算统计量
- `__getitem__` 里的 pad：为了构造固定长度监督样本

---

### 4.4 它算的是什么统计量

后面会 stack 后求：

```python
action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
action_std = all_action_data.std(dim=[0, 1], keepdim=True)
qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
```

这表示它是在：

- episode 维
- 时间维

一起聚合，最后保留每个动作维度 / 状态维度的统计量。

随后还会：

```python
torch.clip(..., 1e-2, np.inf)
```

防止某些维度标准差过小。

---

### 4.5 `get_norm_stats(...)` 的真正作用

这一步最后返回：

```python
return stats, max_action_len
```

所以你应该把这个函数理解成：

> **Dataset 构造前的全局统计准备器**。

没有它，后面 `EpisodicDataset` 就不知道：

- qpos / action 怎么归一化
- 动作样本要 pad 到多长

---

## 5. 主体：`EpisodicDataset.__init__`

位置仍然在：

```text
policy/ACT/utils.py
```

构造函数：

```python
def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len):
    super(EpisodicDataset).__init__()
    self.episode_ids = episode_ids
    self.dataset_dir = dataset_dir
    self.camera_names = camera_names
    self.norm_stats = norm_stats
    self.max_action_len = max_action_len
    self.is_sim = None
    self.__getitem__(0)  # initialize self.is_sim
```

这一段每一项都不是可跳过的。

---

### 5.1 `episode_ids`

这个变量决定这个 Dataset 对象管理哪些 episode 文件。

它不是 timestep id，而是 episode 编号列表，比如：

```python
[0, 3, 5, 7, 9, ...]
```

所以 Dataset 的外层索引语义是：

```text
index -> episode_ids[index] -> 某个 episode_i.hdf5
```

而不是：

```text
index -> 某个固定 timestep 样本
```

---

### 5.2 `dataset_dir`

这是处理后 HDF5 所在目录，比如：

```text
processed_data/sim-beat_block_hammer/demo_clean-50
```

所以 Dataset 的输入源已经不是 assets，也不是原始 collect 产生的缓存文件，而是 ACT 训练使用的统一 HDF5 格式数据。

---

### 5.3 `camera_names`

这决定：

- 读哪些相机
- 相机的排列顺序
- 最终 `image_data[0], image_data[1], ...` 各自代表哪个视角

这一点非常重要，因为后面图像会被 stack 成 `[num_cams, ...]`，相机维是显式存在的。

---

### 5.4 `norm_stats`

这个字段供 `qpos_data` 和 `action_data` 做标准化使用。

图像不走这一套统计量，图像在 Dataset 里只做 `/255.0`。

所以这份实现里有两种不同的归一化逻辑：

- 图像：基础像素缩放
- qpos / action：基于整个数据集统计量的标准化

---

### 5.5 `max_action_len`

这个变量决定训练样本里动作后缀 pad 到多长。

它不是 chunk size，也不是 `num_queries`，而是：

> 整个数据集中最长 action 序列的长度。

后面 `ACTPolicy` 才会再把它裁成 `num_queries == chunk_size`。

---

### 5.6 `self.is_sim = None` 和 `self.__getitem__(0)`

这一组代码值得单独说。

作者的意图看起来是：

> 通过先取一个样本，初始化 `self.is_sim`

但在当前这份实现里，`__getitem__` 内部实际上是：

```python
is_sim = None
...
self.is_sim = is_sim
```

也就是说，当前版本并没有真正从 HDF5 中恢复出 sim / real 标志。

所以这里应理解为：

- 代码保留了这套接口设计意图
- 但在当前实现里，这个逻辑没有真正发挥作用

这类“半残留设计”是读源码时很常见的，你要会识别。

---

## 6. `__len__`：为什么它返回的是 episode 数，不是样本数

```python
def __len__(self):
    return len(self.episode_ids)
```

这句很关键，因为它决定了 DataLoader 外层遍历粒度。

这里的含义是：

> 一个 epoch 中，DataLoader 外层遍历的是 episode 集合，而不是全部 timestep 样本集合。

真正的 timestep 随机性，来自 `__getitem__` 内部的 `start_ts` 采样，而不是 `__len__`。

所以这份 Dataset 的采样策略是：

### 外层
按 episode 枚举

### 内层
在每个 episode 中随机采一个起点时刻

这和把 `(episode_id, timestep)` 预先全部展开成大索引表的 Dataset 设计不同。

---

## 7. `__getitem__` 前半段：先定义“一个样本”是什么

这一段是最核心的。

主框架如下：

```python
def __getitem__(self, index):
    sample_full_episode = False

    episode_id = self.episode_ids[index]
    dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
    with h5py.File(dataset_path, "r") as root:
        is_sim = None
        original_action_shape = root["/action"].shape
        episode_len = original_action_shape[0]
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        qpos = root["/observations/qpos"][start_ts]
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
        if is_sim:
            action = root["/action"][start_ts:]
            action_len = episode_len - start_ts
        else:
            action = root["/action"][max(0, start_ts - 1):]
            action_len = episode_len - max(0, start_ts - 1)
```

一句话先概括它的语义：

> 先按 `index` 选中某个 episode 文件，再在这个 episode 内随机选择一个起点时刻 `start_ts`，只取该时刻的观测（图像 + qpos），再取从该时刻附近开始的一段未来动作后缀作为监督。

---

### 7.1 `sample_full_episode = False`

这一句不能跳过，因为它是个真实功能开关。

当前含义是：

- 默认不从头取整段 episode
- 而是随机采一个局部起点

如果设为 `True`，那下面 `start_ts = 0`，样本就总是从 episode 开头构造。

所以这里反映的是：

> 当前实现默认采用“局部起点采样”的方式训练 ACT。

---

### 7.2 `episode_id -> dataset_path`

```python
episode_id = self.episode_ids[index]
dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
```

说明一个 `Dataset` 索引先对应的是 episode 文件，而不是具体时刻样本。

---

### 7.3 `original_action_shape` 和 `episode_len`

```python
original_action_shape = root["/action"].shape
episode_len = original_action_shape[0]
```

为什么先看 action 的长度？

因为这份 Dataset 不是纯图像集，而是“观测 + 动作监督”联合样本。这里用 action 序列长度来定义 episode 的时间长度，是合理的。

---

### 7.4 `start_ts`

```python
start_ts = np.random.choice(episode_len)
```

这一步决定：

- 当前观测取哪一个时间点
- 未来动作后缀从哪里开始切

它的设计意图是：

> 让同一条 episode 能在不同 epoch 中提供不同时间位置的训练样本，而不是只训练 episode 开头的状态。

---

### 7.5 当前观测中的 `qpos`

```python
qpos = root["/observations/qpos"][start_ts]
```

说明输入的机器人状态只取当前时刻一帧，而不是状态序列。

---

### 7.6 当前观测中的图像：`image_dict`

```python
image_dict = dict()
for cam_name in self.camera_names:
    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
```

这一步的意思是：

- 对每个相机
- 读取该相机在 `start_ts` 时刻的一张图像

如果 HDF5 里某个相机字段的 shape 是：

```text
[T, H, W, C]
```

那么取 `[start_ts]` 后得到的是：

```text
[H, W, C]
```

所以：

> 当前 ACT 实现读图时，不是取整段视频，而是只取**单时刻多视角图像**。

---

### 7.7 action 切片：当前实现实际走哪条分支

```python
if is_sim:
    action = root["/action"][start_ts:]
    action_len = episode_len - start_ts
else:
    action = root["/action"][max(0, start_ts - 1):]
    action_len = episode_len - max(0, start_ts - 1)
```

因为当前 `is_sim = None`，它在 Python 里会被当作 False。也就是说，**这份代码实际始终走 `else` 分支**。

真正执行的是：

```python
action = root["/action"][max(0, start_ts - 1):]
```

这说明当前实现用了一个时间对齐 hack：

> 用 `start_ts - 1` 附近开始的动作后缀，去和当前观测对齐。

这不是通用 ACT 理论规定，而是这份仓库的实现细节。

---

## 8. `__getitem__` 后半段：pad、图像堆叠、tensor 化、归一化

接着看后半段：

```python
self.is_sim = is_sim

padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
padded_action[:action_len] = action
is_pad = np.ones(self.max_action_len, dtype=bool)
is_pad[:action_len] = 0

all_cam_images = []
for cam_name in self.camera_names:
    all_cam_images.append(image_dict[cam_name])
all_cam_images = np.stack(all_cam_images, axis=0)

image_data = torch.from_numpy(all_cam_images)
qpos_data = torch.from_numpy(qpos).float()
action_data = torch.from_numpy(padded_action).float()
is_pad = torch.from_numpy(is_pad).bool()

image_data = torch.einsum("k h w c -> k c h w", image_data)
image_data = image_data / 255.0
action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

return image_data, qpos_data, action_data, is_pad
```

---

### 8.1 `self.is_sim = is_sim`

当前版本里，这句的实际效果仍然只是把 `None` 赋回 `self.is_sim`。

所以它保留了接口，但没有真正承担 sim/real 分流功能。

---

### 8.2 `padded_action`

```python
padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
padded_action[:action_len] = action
```

这里的作用是：

- 先创建一个长度固定为 `max_action_len` 的全零动作数组
- 再把真实动作后缀填进前半段

这样每个样本的 `action_data` 长度都被统一了。

这一步非常重要，因为没有它，后面 DataLoader 默认 collate 根本无法把不同样本的 action 序列 stack 成 batch。

---

### 8.3 `is_pad`

```python
is_pad = np.ones(self.max_action_len, dtype=bool)
is_pad[:action_len] = 0
```

这里：

- 前半段真实动作位置：`False`
- 后半段补零位置：`True`

这个布尔 mask 后面会在 [[RoboTwin/ACT/act_policy|act_policy]] 里用于屏蔽 padding 部分的动作损失。

---

### 8.4 多相机图像堆叠

```python
all_cam_images = []
for cam_name in self.camera_names:
    all_cam_images.append(image_dict[cam_name])
all_cam_images = np.stack(all_cam_images, axis=0)
```

这三行是图像样本构造的核心。

每个 `image_dict[cam_name]` 都是一张：

```text
[H, W, C]
```

`np.stack(axis=0)` 后变成：

```text
[num_cams, H, W, C]
```

所以：

> 这份实现把多相机图像保留为显式的“相机维”，而不是提前拼成一张大图。

---

### 8.5 tensor 化

```python
image_data = torch.from_numpy(all_cam_images)
qpos_data = torch.from_numpy(qpos).float()
action_data = torch.from_numpy(padded_action).float()
is_pad = torch.from_numpy(is_pad).bool()
```

这一步把 NumPy 数组正式变成训练中使用的 tensor。

注意这里：

- `image_data` 此时 shape 还是 `[num_cams, H, W, C]`
- 只是数据类型换成了 PyTorch tensor

---

### 8.6 图像维度重排：`einsum("k h w c -> k c h w")`

```python
image_data = torch.einsum("k h w c -> k c h w", image_data)
```

变换前：

```text
[num_cams, H, W, C]
```

变换后：

```text
[num_cams, C, H, W]
```

作用是把 channel-last 图像改成 PyTorch CNN 习惯的 channel-first 图像。

虽然写成 `einsum`，但本质上只是维度置换，不是在做复杂运算。

---

### 8.7 归一化

```python
image_data = image_data / 255.0
action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
```

这里一定要并列理解三类数据：

### 图像
只做 `/255.0`，把像素从 `[0,255]` 缩放到 `[0,1]`。

### qpos
减均值除标准差。

### action
减均值除标准差。

所以 Dataset 内部其实有三套不同的数值处理：

- 图像：基础像素缩放
- qpos：标准化
- action：标准化

---

### 8.8 `__getitem__` 的返回值到底意味着什么

最后返回：

```python
return image_data, qpos_data, action_data, is_pad
```

所以一个样本的真正语义是：

```text
当前时刻多相机图像 image_data
+ 当前时刻机器人状态 qpos_data
+ 从当前时刻附近开始的一段未来动作后缀 action_data
+ 这段后缀中哪些位置是 padding 的 is_pad
```

其中图像部分的最终 shape 是：

```text
[num_cams, C, H, W]
```

---

## 9. `load_data(...)`：DataLoader 是怎么被构造出来的

函数原型：

```python
def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
```

它主要做四件事：

1. train / val 划分
2. 计算 `norm_stats` 和 `max_action_len`
3. 构造 train / val dataset
4. 构造 train / val dataloader

---

### 9.1 train / val 划分是 episode 级别的

```python
train_ratio = 0.8
shuffled_indices = np.random.permutation(num_episodes)
train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
val_indices = shuffled_indices[int(train_ratio * num_episodes):]
```

这里划分的是 episode，而不是 timestep。

也就是说：

- 某个 episode 要么进 train
- 要么进 val

这样可以避免同一条轨迹不同时间步同时跑到训练集和验证集里，减少泄漏。

---

### 9.2 调 `get_norm_stats(...)`

```python
norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)
```

这一步把前面讲的全局统计量和最大长度准备好。

---

### 9.3 构造 train_dataset / val_dataset

```python
train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_action_len)
val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_action_len)
```

这说明：

- train / val 用的是同一个 Dataset 类
- 区别只在它们持有的 episode 子集不同

---

### 9.4 构造 DataLoader

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size_train,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    prefetch_factor=1,
)
```

验证集同理。

这里逐项理解：

#### `batch_size`
把单样本四元组 stack 成 batch 四元组。图像从：

```text
[num_cams, C, H, W]
```

变成：

```text
[B, num_cams, C, H, W]
```

#### `shuffle=True`
打乱的是外层 episode 访问顺序；再结合 `__getitem__` 内部随机 `start_ts`，就有两层随机性。

#### `pin_memory=True`
提升 CPU -> GPU 拷贝效率。

#### `num_workers=1`
说明这份实现采用的是比较保守、稳定的加载配置。

#### `prefetch_factor=1`
预取也设得比较保守。

---

### 9.5 为什么 DataLoader 可以用默认 collate 直接拼 batch

虽然代码里没有自己写 `collate_fn`，但默认 collate 能成功工作，背后是有条件的。

DataLoader 取到若干个 `__getitem__` 返回值后，会按位置自动 stack：

- `image_data`: `[num_cams, C, H, W]` -> `[B, num_cams, C, H, W]`
- `qpos_data`: `[state_dim]` -> `[B, state_dim]`
- `action_data`: `[max_action_len, action_dim]` -> `[B, max_action_len, action_dim]`
- `is_pad`: `[max_action_len]` -> `[B, max_action_len]`

这里最关键的前提是：

> `action_data` 和 `is_pad` 已经在 `__getitem__` 中 pad 到了统一长度。

所以一定要把这条链记住：

```text
get_norm_stats(...) -> max_action_len
->__getitem__() pad action / is_pad
-> DataLoader default collate 才能成功 stack
```

---

## 10. `imitate_episodes.py`：dataloader 怎么接进训练主线

如果 `utils.py` 负责“数据怎么来”，
那么 `imitate_episodes.py` 负责“训练脚本怎么把它用起来”。

位置：

```text
policy/ACT/imitate_episodes.py
```

---

### 10.1 `main(...)` 中 task config 提供 dataset 参数

训练脚本会先从 task config 里拿：

```python
dataset_dir = task_config["dataset_dir"]
num_episodes = task_config["num_episodes"]
episode_len = task_config["episode_len"]
camera_names = task_config["camera_names"]
```

这里 `camera_names` 很重要，因为它最终会传进 `load_data(...)`，从而决定 Dataset 取哪些相机、相机顺序是什么。

---

### 10.2 调 `load_data(...)`

```python
train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)
```

这句就是 dataset / dataloader 正式接入训练主线的入口。

---

### 10.3 保存 `dataset_stats.pkl`

```python
stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
with open(stats_path, "wb") as f:
    pickle.dump(stats, f)
```

这一块值得记，因为这说明训练阶段计算出来的归一化统计量，不只是临时用一下。

它后面在部署 / 推理时也会被读取，用来做：

- `qpos` 归一化
- 动作反归一化

也就是说，Dataset 这一步产出的统计量，会影响整个后续使用链。

---

### 10.4 `forward_pass(...)`

```python
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)
```

这里有两个重点：

#### 第一
DataLoader 返回的 batch 结构被原样接住，没有再改 shape。

#### 第二
这里只做了 `.cuda()`，说明 Dataset / DataLoader 输出的张量形状已经完全符合 policy 接口预期。

所以你可以把 `forward_pass(...)` 看成：

> 数据管线和模型管线的交界点。

---

## 11. `act_policy.py`：为什么图像处理还没结束

前面只讲到 Dataset 内部图像做了 `/255.0`。但这还不是最终送入模型的图像。

位置：

```text
policy/ACT/act_policy.py
```

这里和图像最相关的有两块：

1. `resize_image_tensor(...)`
2. `ACTPolicy.__call__(...)` 中对图像的处理

---

### 11.1 `resize_image_tensor(...)`

函数会判断：

```python
if image.dim() == 5:
    b, ncam, c, h, w = image.shape
    image = image.reshape(b * ncam, c, h, w)
    image = F.interpolate(image, size=size, mode="bilinear", align_corners=False, antialias=True)
    image = image.reshape(b, ncam, c, size[0], size[1])
```

这说明它明确预期 DataLoader 输出的图像 batch 是：

```text
[B, num_cams, C, H, W]
```

而 `F.interpolate` 只接受标准 4 维图像 batch `[N, C, H, W]`，所以这里先把：

- batch 维 `B`
- 相机维 `num_cams`

合并成一个大 batch 维 `B * num_cams`，resize 之后再 reshape 回去。

这个实现非常值得记，因为它说明：

> 多相机维在整个数据流里一直被显式保留，只有在做图像插值时才临时和 batch 维合并。

---

### 11.2 `ACTPolicy.__call__(...)` 中的图像预处理

训练和推理进入 policy 后，都会先做：

```python
image = resize_image_tensor(image, self.target_image_size)
image = normalize(image)
```

其中 `normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

所以图像预处理实际上被分成了两层：

### Dataset 内
- HDF5 读图
- 多相机 stack
- HWC -> CHW
- `/255.0`

### Policy 内
- resize 到目标分辨率
- ImageNet mean/std 标准化

这也解释了为什么 Dataset 里图像只做 `/255.0`，而没有继续做 ImageNet normalize：

> 因为模型输入规格对齐这件事，被作者有意放到了 `ACTPolicy` 层。

---

## 12. 到这里，图像张量的 shape 是怎么一路变化的

这是你之后复习时最值得背的一段。

### HDF5 中单相机单时刻图像

```text
[H, W, C]
```

### Dataset 中多相机 stack 后

```text
[num_cams, H, W, C]
```

### Dataset 中换通道顺序后

```text
[num_cams, C, H, W]
```

### DataLoader collate 成 batch 后

```text
[B, num_cams, C, H, W]
```

### Policy 中为了 resize 临时 reshape

```text
[B * num_cams, C, H, W]
```

### resize 后 reshape 回去

```text
[B, num_cams, C, H_target, W_target]
```

之后再做 ImageNet normalize，shape 不变，再进入视觉 backbone。

---

## 13. 这一整块在 ACT 任务定义中的意义

理解到这里，还要再往上提一层。

这份 Dataset / DataLoader 设计之所以是现在这样，不是任意的，而是因为 ACT 的任务定义是：

> 给定当前时刻观测，预测未来一段动作 chunk。

所以它才会选择：

- 图像：只取当前时刻单帧，但保留多视角
- qpos：只取当前时刻状态
- action：取未来动作后缀，再 pad
- policy：再裁成 `num_queries == chunk_size`

因此，这套数据管线其实和 [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]、[[RoboTwin/ACT/train-and-infer|训练与推理流程]] 是强耦合的。

---

## 14. 复习时最值得抓住的 8 个点

1. Dataset / DataLoader 主体确实主要在 `utils.py`
2. 但完整链要连到 `imitate_episodes.py` 和 `act_policy.py`
3. `__len__` 返回的是 episode 数，不是 timestep 数
4. `__getitem__` 在 episode 内部随机采 `start_ts`
5. 图像输入是**单时刻多相机图像**，不是视频片段
6. 图像在 Dataset 中被构造成 `[num_cams, C, H, W]`
7. DataLoader 默认 collate 之所以能工作，是因为 action / is_pad 已先 pad 到固定长度
8. 真正进入模型前，图像还会在 `ACTPolicy` 中再做 resize 和 ImageNet normalize

---

## 15. 一句话总结

> 在 RoboTwin 的 ACT 实现里，`utils.py` 负责把处理后 HDF5 中的单时刻多相机图像、当前 qpos、未来动作后缀和 pad mask 组织成统一长度的单样本；`load_data(...)` 再用默认 DataLoader 把这些样本堆成 batch；`imitate_episodes.py` 负责把 batch 接入训练主线；而 `act_policy.py` 则继续把图像 batch 对齐到视觉 backbone 所需的输入规格。整个 dataset / dataloader 设计，本质上是在为“当前观测 -> 未来动作 chunk”的 ACT 任务定义服务。
