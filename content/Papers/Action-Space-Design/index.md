---
title: Action Space Design
---

# Action Space Design

这一类论文重点关注：

- policy 输出到底定义在什么空间
- absolute / delta 有什么差别
- joint / task space 如何权衡
- action chunking、horizon、control interface 会怎样影响 learnability 和 execution stability

这类论文很重要，因为它们讨论的不是某个模型结构的小改动，而是：

> 神经网络输出如何真正变成机器人可执行控制命令。

## 当前论文

- [[Papers/Action-Space-Design/Demystifying-Action-Space-Design-for-Robotic-Manipulation-Policies/index|Demystifying Action Space Design for Robotic Manipulation Policies]]

## 这一类论文适合怎么读

建议优先抓住三件事：

1. 作者到底把动作空间拆成了哪些维度
2. 这些维度对 learnability / stability / generalization 各有什么影响
3. 最后的结论是不是条件化结论，而不是“谁永远最好”的口号
