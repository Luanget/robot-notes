---
title: ACT
---

# ACT

这部分笔记围绕 RoboTwin 仓库中的 ACT 策略实现展开。

现在的阅读策略是：

- **一篇主线页**负责讲清 ACT 最核心的链路
- 其它页面负责补细节，而不再重复讲主线

---

## 建议阅读顺序

### 1. 先看主线

- [[RoboTwin/ACT/act-overall-dataflow|ACT 核心链路]]
  - 这篇是总入口，建议最先看。
  - 它已经合并了原来的“整体数据流”和“训练与推理流程”。

### 2. 再看输入与样本来源

- [[RoboTwin/ACT/dataset-and-dataloader|Dataset 与 DataLoader]]
  - 负责解释样本是怎么从 HDF5 变成 batch 的，图像 / qpos / action / is_pad 从哪里来。

### 3. 再看模型主体

- [[RoboTwin/ACT/detr_vae|DETR-VAE 主体结构]]
- [[RoboTwin/ACT/transformer-dataflow|Transformer 数据流]]
- [[RoboTwin/ACT/act_policy|ACT Policy 与损失组织]]

### 4. 最后看概念页与源码定位

- [[RoboTwin/ACT/concepts/index|ACT 概念索引]]
- [[RoboTwin/ACT/code-file-map|ACT 代码文件地图]]

---

## 这套笔记现在怎么用

如果你是为了复习而不是第一次读代码，建议只抓下面这条线：

```text
Dataset / DataLoader
-> ACT 核心链路
-> Transformer / DETR-VAE
-> Policy 与 loss
-> concepts 回查
```

这样不会在重复页面里来回绕。
