---
title: image tokens
---

# image tokens

## 1. image tokens 是什么

在 ACT 里，`image tokens` 指的是由相机图像经过视觉 backbone 编码后，再送入主 transformer 的那一串视觉 token。

在 RoboTwin 这份实现里，它们不是直接把原始图像切 patch 得到的，而是经过下面这条路径：

```text
image -> backbone -> feature map -> 1x1 conv 投影 -> flatten -> image tokens
```

所以 image tokens 更准确地说是：

> **backbone 输出特征图上的空间位置 token**，不是原始像素块本身。

---

## 2. 它们在 ACT 里负责什么

image tokens 负责提供外部视觉环境信息，例如：

- 目标物体的位置与外观
- 场景中的障碍、桌面、容器等布局
- 机器人末端与物体的相对空间关系
- 多个相机视角下的补充线索

如果没有 image tokens，ACT 只能依赖：

- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]：机器人自身状态
- [[RoboTwin/ACT/concepts/latent-token|latent token]]：动作模式条件

但就无法知道“当前外部世界长什么样”。

因此 image tokens 是动作生成的视觉条件来源。

---

## 3. 在你的仓库里，image tokens 是怎么来的

### 3.1 每个相机都先过 backbone

在 `policy/ACT/detr/models/detr_vae.py` 中：

```python
for cam_id, cam_name in enumerate(self.camera_names):
    features, pos = self.backbones[0](image[:, cam_id])
    features = features[0]
    pos = pos[0]
    all_cam_features.append(self.input_proj(features))
    all_cam_pos.append(pos)
```

这里有几个实现级事实需要记住：

#### 第一，所有相机共享同一个 backbone 实例

代码里用的是：

```python
self.backbones[0](image[:, cam_id])
```

而不是 `self.backbones[cam_id]`。  
所以这份实现中，多路相机共享同一套视觉编码器参数。

#### 第二，取的是 backbone 最后一层 feature

`features[0]`、`pos[0]` 表示取返回列表中的最后一级特征。

#### 第三，会再经过 `input_proj`

```python
self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
```

这一步把 backbone 的通道数投到 transformer 使用的 `hidden_dim`。

---

## 4. 为什么还要做 `input_proj`

视觉 backbone 输出的通道维通常不等于 transformer 的隐藏维度。  
但主 transformer 里的 token 需要统一维度。

所以这一步：

```python
1x1 conv: C_backbone -> hidden_dim
```

就是在做视觉 token 的维度对齐。

它的作用不是提取新空间结构，而是：

- 保留 feature map 的空间布局
- 只改通道维
- 让后续 flatten 后的 image tokens 能和 latent / proprio 在同一 hidden space 中交互

---

## 5. 多相机特征是如何合并的

这是你仓库实现里一个非常值得记的细节。

处理完每个相机后，并不是先各自独立送进 transformer，而是：

```python
src = torch.cat(all_cam_features, axis=3)
pos = torch.cat(all_cam_pos, axis=3)
```

也就是：

- 沿宽度维拼接所有相机的 feature map
- 对应的位置编码也沿宽度维拼接

这意味着从主 transformer 的视角看：

> 多相机视觉信息会先组成一个“更宽的统一特征图”，之后再 flatten 成一串 image tokens。

所以 image tokens 虽然来自多个相机，但最终会进入一个统一视觉 token 序列。

---

## 6. image tokens 真正形成的时刻

在 `detr_vae.py` 中还只是 feature map。  
真正变成 transformer 输入 token 序列，是在 `transformer.py` 里完成的。

从逻辑上讲，这一步相当于：

```text
(bs, hidden_dim, H, W_total)
   -> flatten spatial dims
(H * W_total, bs, hidden_dim)
```

也就是说，feature map 上每个空间位置都会变成一个 token。  
这些 token 加上位置编码后，就构成了 image tokens。

所以 image token 的“一个 token”对应的不是整张图，而是：

> feature map 上某个空间位置的视觉表示。

---

## 7. image tokens 和 patch token 的关系

如果你熟悉 ViT，很容易把 image token 理解成“patch token”。

在 ACT 这份实现里，更准确的说法是：

- 它们功能上类似视觉 token
- 但来源不是直接对原图做 ViT patch embedding
- 而是 CNN backbone 输出的 feature map 位置

所以别简单地把它们等同成“原图 patch”。  
更贴实现的说法是：

> image tokens 是 backbone 特征图经投影并 flatten 后得到的空间视觉 token。

---

## 8. image tokens 和位置编码的关系

图像 token 如果没有位置信息，transformer 只知道“有哪些视觉特征”，却不知道它们来自哪里。

因此每路相机经过 backbone 时，除了 `features` 以外，还会返回 `pos`。  
后续多相机的 `pos` 也会一起拼接。

这意味着 image tokens 进入 transformer 时，不只是内容向量，还附带了空间位置信息。

这样 encoder 才能理解：

- 哪些 token 来自画面左上角
- 哪些 token 来自右侧手腕相机区域
- 哪些视觉位置彼此相邻

所以 image tokens 实际是：

```text
视觉内容表示 + 空间位置编码
```

---

## 9. image tokens 在 encoder 中会发生什么

进入主 transformer encoder 后，image tokens 不再只是“本地视觉特征”，而会和：

- [[RoboTwin/ACT/concepts/latent-token|latent token]]
- [[RoboTwin/ACT/concepts/proprio-token|proprio token]]
- 其他 image tokens

一起做 self-attention 融合。

这一步的结果是：

- 图像 token 不再只知道局部外观
- 还会吸收 robot state 和 latent 条件信息
- 最终成为 [[RoboTwin/ACT/concepts/memory|memory]] 的一部分

所以 decoder 后面读到的，不是“原始图像 token”，而是“融合后的视觉相关 memory token”。

---

## 10. image tokens 和 decoder 的关系

decoder 并不会直接输入图像，也不会直接操作 feature map。  
它只会通过 cross-attention 访问 encoder memory。

因此 image tokens 对动作输出的影响路径是：

1. 图像 -> image tokens
2. image tokens 与 latent / proprio 一起进入 encoder
3. encoder 输出 memory
4. [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]] 从 memory 中读取相关视觉信息
5. 最终经 [[RoboTwin/ACT/concepts/action-head|action head]] 输出动作 chunk

也就是说，image tokens 负责提供“视觉条件”，不是直接输出动作。

---

## 11. 多相机为什么重要

RoboTwin 这里使用多路相机，例如头部视角、左右手附近视角。  
这么做的好处是：

- 头部视角适合提供全局场景布局
- 手边视角更容易看到抓取接触细节
- 不同视角互补，减少遮挡问题

而当前实现采取的是“早期统一”：

- 每个相机先各自过共享 backbone
- 然后在特征图阶段拼接
- 再一起进入同一个主 transformer

所以主 transformer 能同时在统一序列中看到多视角视觉信息。

---

## 12. 看源码时最该记住哪几行

### 构造视觉投影

```python
self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
```

### 每路相机提特征

```python
features, pos = self.backbones[0](image[:, cam_id])
features = features[0]
pos = pos[0]
```

### 维度对齐并多相机拼接

```python
all_cam_features.append(self.input_proj(features))
all_cam_pos.append(pos)
src = torch.cat(all_cam_features, axis=3)
pos = torch.cat(all_cam_pos, axis=3)
```

这几句基本就对应了 image tokens 的整条生成链。

---

## 13. 一句话总结

> image tokens 是多路相机图像经过共享 backbone 编码、再经 1x1 conv 投影并 flatten 后得到的视觉 token；
> 它们提供外部场景信息，与 latent token 和 proprio token 一起进入主 transformer encoder，形成后续动作生成所依赖的 memory。
