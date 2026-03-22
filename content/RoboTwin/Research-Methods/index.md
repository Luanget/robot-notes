---
title: 研究能力升级
---

# 研究能力升级

这组笔记不是在补卷积、Transformer、多头注意力这些基础内容，而是面向当前阶段：

- 已经能较深入理解 ACT / DP
- 已经在 RoboTwin 中真实修改训练链、数据链、评测链
- 已经能读较复杂的 policy / dataset / workspace / loss 代码
- 下一步想提升的，不是“知道更多知识点”，而是更高一层的研究能力

这组笔记的目标是整理三类更关键的能力：

1. [[RoboTwin/Research-Methods/problem-definition|问题定义能力]]
2. [[RoboTwin/Research-Methods/causal-diagnosis|因果诊断能力]]
3. [[RoboTwin/Research-Methods/research-expression|研究表达能力]]

---

## 为什么单独建这一组笔记

`RoboTwin/ACT/` 和后续的 `DP / action space / backbone` 相关笔记，更偏向：

- 代码实现
- 数据流 / 张量流
- 模型结构
- 训练与推理流程

而这一组笔记更偏向：

- 一个问题值不值得做
- 一个实验现象该怎么解释
- 一个方法该怎么被写成论文命题
- 一个失败该如何复盘为可迁移经验

所以它和前面的笔记不是重复关系，而是更高一层的“研究方法层”。

---

## 建议阅读顺序

### 1. 先看总览

- [[RoboTwin/Research-Methods/problem-definition|问题定义能力]]
- [[RoboTwin/Research-Methods/causal-diagnosis|因果诊断能力]]
- [[RoboTwin/Research-Methods/research-expression|研究表达能力]]

### 2. 再看工具页

- [[RoboTwin/Research-Methods/research-question-card|研究问题卡模板]]
- [[RoboTwin/Research-Methods/failure-review-template|失败复盘模板]]
- [[RoboTwin/Research-Methods/experiment-checklist|实验设计检查清单]]

### 3. 最后回到当前项目里应用

建议把这组笔记和下面这些内容来回对应：

- `RoboTwin/ACT/`
- `Papers/Action-Space-Design/`
- 你当前在 RoboTwin 中做的 `DP_geoaux / action space / training chain` 改动

---

## 这组笔记最重要的一句话

如果已经过了“基础知识学习”阶段，那么下一步真正该练的，不是再会更多公式，而是：

> 看到一个现象，能把它定义成问题；
> 做出一组实验，能把它解释成机制；
> 想到一个方法，能把它表达成论文主张。
