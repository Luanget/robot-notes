---
title: ACT 概念索引
---

# ACT 概念索引

这一组页面不是按源码文件拆，而是按“在 ACT 主线里反复出现、必须单独搞清楚的核心概念”来拆。

如果你的目标是：

- 不看源码也能迅速回忆 ACT 主线
- 看到某个词就能马上知道它在整条数据流中的位置
- 理清 encoder / decoder / latent / action chunk 之间的关系

那么建议按下面顺序复习。

---

## 1. 先看输入侧三个条件 token

### [[RoboTwin/ACT/concepts/latent-token|latent token]]

回答：

- latent token 从哪里来
- 为什么训练时来自 posterior，推理时直接用零向量
- 它在 ACT 中承担什么高层动作模式作用

这是理解 ACT 为什么是 CVAE + transformer 的关键。

### [[RoboTwin/ACT/concepts/proprio-token|proprio token]]

回答：

- 当前机器人状态 `qpos` 如何变成 token
- 为什么不能只靠图像代替 robot state
- 它如何作为 encoder 条件输入参与动作生成

### [[RoboTwin/ACT/concepts/image-tokens|image tokens]]

回答：

- 多路相机图像如何经过 backbone 变成视觉 token
- 为什么还需要 `input_proj`
- 多相机特征在这份实现里是怎么合并的

---

## 2. 再看 encoder 输出侧的统一条件表示

### [[RoboTwin/ACT/concepts/memory|memory]]

回答：

- memory 到底是什么
- 为什么它不是单个 pooled 向量，而是一整段 encoder 输出序列
- decoder 为什么必须从 memory 中读信息

这是把“输入条件”与“输出动作槽位”接起来的核心概念。

---

## 3. 再看 decoder 端如何把条件变成动作

### [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]

回答：

- 为什么一个 query 对应一个动作槽位
- 为什么 decoder 输入是零初始化 `tgt`
- self-attention / cross-attention 如何让 query 变成动作表示

这是理解 ACT decoder 的关键页。

### [[RoboTwin/ACT/concepts/action-head|action head]]

回答：

- decoder 输出如何最终映射成动作向量
- 为什么这里只用一个线性层
- 它和外层 L1 loss、chunk 动作执行有什么关系

---

## 4. 如果你想按“主线”而不是“概念”复习

建议直接跳到这些页面：

- [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]：总览页，适合快速回忆 ACT 从输入到输出的全流程
- [[RoboTwin/ACT/train-and-infer|train-and-infer]]：训练 / 推理两条数据流对照页
- [[RoboTwin/ACT/detr_vae|detr_vae]]：模型结构总装页
- [[RoboTwin/ACT/act_policy|act_policy]]：外层 loss、chunk 调用和 temporal aggregation
- [[RoboTwin/ACT/transformer-dataflow|ACT 中 Transformer 数据流转]]：主 transformer 内部主线

---

## 5. 一套最推荐的复习顺序

如果你要在最短时间内把 ACT 重新回忆起来，我建议按下面顺序：

1. [[RoboTwin/ACT/act-overall-dataflow|ACT 整体数据流]]
2. [[RoboTwin/ACT/train-and-infer|train-and-infer]]
3. [[RoboTwin/ACT/concepts/latent-token|latent token]]
4. [[RoboTwin/ACT/concepts/query-embeddings|query embeddings]]
5. [[RoboTwin/ACT/concepts/memory|memory]]
6. [[RoboTwin/ACT/detr_vae|detr_vae]]
7. [[RoboTwin/ACT/act_policy|act_policy]]
8. 再补 [[RoboTwin/ACT/concepts/proprio-token|proprio token]]、[[RoboTwin/ACT/concepts/image-tokens|image tokens]]、[[RoboTwin/ACT/concepts/action-head|action head]]

这样复习的好处是：

- 先抓总线
- 再补训练/推理差异
- 然后把最关键的概念点逐个钉住
- 最后再回到模块级实现

---

## 6. 一句话总结

> 这组概念页的作用，不是替代主线笔记，而是把 ACT 中最容易混淆、最值得单独记忆的几个核心对象拆开讲清楚；
> 这样你以后看到 latent token、query embeddings、memory 这些词时，能立刻把它们放回整条 ACT 数据流中定位。
