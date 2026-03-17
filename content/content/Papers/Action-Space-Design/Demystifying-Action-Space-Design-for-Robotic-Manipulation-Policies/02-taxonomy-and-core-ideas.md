---
title: Taxonomy 与核心概念
---

# Taxonomy 与核心概念

这一篇只讲概念骨架。它的目标不是复述实验，而是把：

- spatial abstraction
- temporal abstraction
- action chunking
- configuration manifold

这些核心概念讲清楚。

---

## 1. Spatial Abstraction：空间抽象

空间抽象决定的是：

> 神经网络到底负责到哪一层，后面的事交给底层控制器做。

作者主要比较两种最常见的形式。

### Joint Space
直接预测关节位置或关节增量。

#### 优点

- 不依赖 IK
- 部署链条更短
- 数值执行更稳

#### 难点

- 策略必须隐式学会机器人运动学结构
- 要从视觉输入直接回归到复杂的 joint configuration space
- 这个空间常常是非线性、结构化、甚至多模态的

### Task Space / EE
直接预测末端执行器的位置 / 姿态。

#### 优点

- 更符合视觉中的物体-手几何关系
- 学习目标更有几何语义
- 在弱模型或迁移场景下通常更自然

#### 难点

- 最终还要通过 IK 才能变成 joint commands
- IK 可能带来数值奇异性、误差累积和鲁棒性问题

---

## 2. 我对 Joint vs EE 的个人理解

我在读这篇论文时，觉得下面这个理解很有帮助：

- `EE / task-space` 更像是把一部分“从任务语义到机器人执行”的难度交给了已知几何结构和 IK 求解器
- `joint-space` 则是把这部分难度留在学习器内部，让策略自己学

所以：

- task-space 更像“好学”
- joint-space 更像“好执行”

这正好对应论文所说的 trade-off：

> learning alignment vs execution robustness

---

## 3. Temporal Abstraction：时间抽象

时间抽象关注的是：

> 网络输出的是目标本身，还是相对于当前状态的变化量。

### Absolute
直接预测目标状态。

#### 特点

- 更有 global grounding（全局锚定）
- 长 horizon 下更有全局一致性

#### 难点

- 需要从视觉直接推全局目标
- 学习问题更难

### Delta
直接预测相对变化量。

#### 特点

- 学习目标更局部
- 更符合“下一步修正一点”的闭环直觉
- 更适合现代 imitation learning

#### 难点

- 如果实现方式不合适，容易漂移
- 对 horizon 更敏感

---

## 4. Action Chunking：动作分块

现代策略往往不是一次只预测一步动作，而是一次预测未来一段动作序列。

这会引入两个关键问题：

### 问题 1：delta 的定义不再唯一
一旦进入 chunking，就必须区分：

- **step-wise delta**：每一步相对前一步累加
- **chunk-wise delta**：整个 chunk 里每一步都相对 chunk 起点定义

### 问题 2：horizon `k` 不能独立调
因为：

- delta 更适合短 horizon 来快速纠偏
- absolute 更适合长 horizon 来维持全局一致性

---

## 5. 什么是 configuration manifold（构型流形）

这是我一开始最容易模糊的概念，所以单独记一下。

### 它不是什么

- 它不是 FK 本身
- 也不是 IK 本身

### 它是什么

它指的是：

> 机器人所有合理关节构型形成的那个复杂、非线性、常常多模态的结构空间。

更直观点说：

- task-space 中“手去哪里”通常更直观
- joint-space 中“每个关节该怎么摆”则可能存在多个合理解
- 这些好解在 joint-space 中不一定挨在一起，而可能分布在不同区域

因此，joint-space 学习难，不是因为输出维度多几个，而是因为模型必须从视觉中直接找到这个复杂空间里的正确位置。

---

## 6. 为什么强生成模型更适合 Joint Space

我在阅读时形成的一个重要理解是：

- 同一个末端目标可能对应多个合理 joint 解
- 这会让 joint-space 监督呈现出多模态特征
- 简单 MSE 回归更容易受“模式平均”影响
- flow matching / diffusion 这类强生成模型更擅长表示复杂多峰分布

因此，当论文说 **flow matching 对 joint-space 尤其有效** 时，我的理解是：

> joint-space 的问题不是它没信息，而是它的分布更复杂、更难被简单回归模型正确表达。

---

## 7. 这一篇最终该记住什么

只记住一句也可以：

> task-space 更好学，joint-space 更好执行；absolute 更全局，delta 更局部；chunking 会进一步改变这些设计的含义。
