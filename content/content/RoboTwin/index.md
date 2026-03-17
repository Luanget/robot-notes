---
title: RoboTwin
---

# RoboTwin

这里整理我围绕 RoboTwin 仓库所做的学习笔记。

目前这套笔记的重点还在 **ACT policy** 这条线，也就是 RoboTwin 中基于 `policy/ACT/` 的模仿学习策略实现。后面如果继续扩展，可以把其他 policy、数据处理流程、任务配置、评测流程也逐步整理进来。

## 当前重点：ACT

入口页：[[RoboTwin/ACT/index|ACT 学习笔记]]

这部分主要对应 RoboTwin 仓库里的这些位置：

- `policy/ACT/act_policy.py`
- `policy/ACT/imitate_episodes.py`
- `policy/ACT/detr/models/detr_vae.py`
- `policy/ACT/detr/models/transformer.py`
- `policy/ACT/detr/models/backbone.py`

如果当前目标是理解 ACT 在 RoboTwin 里的实现，可以直接进入：[[RoboTwin/ACT/index|ACT 学习笔记]]。

## 我为什么先整理 ACT

ACT 是 RoboTwin 中一条非常适合深入学习的主线，因为它把下面几类内容连在了一起：

- 多相机图像输入
- 机器人本体状态输入
- transformer encoder-decoder 主干
- CVAE 风格 latent 建模
- chunk 动作预测
- policy 外层训练与推理逻辑

因此，只要把 ACT 这条线学通，就已经能覆盖很多机器人模仿学习代码里常见的核心结构。

## ACT 笔记内部阅读建议

如果你是第一次回到这套笔记，建议从这里进入：

1. [[RoboTwin/ACT/index|ACT 学习笔记入口]]
2. [[RoboTwin/ACT/01-act-overall-dataflow|ACT 整体数据流]]
3. [[RoboTwin/ACT/02-train-and-infer|训练与推理流程]]
4. [[RoboTwin/ACT/concepts/index|ACT 关键概念索引]]

这样可以先把主线抓住，再按概念跳转补细节。

## 后续可继续扩展的方向

目前这套站点还可以继续往下扩展成更完整的 RoboTwin 学习地图，比如：

- 数据采集与数据处理流程
- 训练脚本与配置项解释
- 不同 policy 之间的对比
- 任务定义、场景配置与评测流程
- 仓库整体目录导航

所以这一页的作用更像是顶层导航页：

> 先告诉自己“现在这套笔记重点在 RoboTwin 的哪一块”，再从具体专题页进入。
