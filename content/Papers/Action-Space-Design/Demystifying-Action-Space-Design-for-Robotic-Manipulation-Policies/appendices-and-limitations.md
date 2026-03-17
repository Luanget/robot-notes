---
title: 附录补充、局限与未来方向
---

# 附录补充、局限与未来方向

这一篇不再重新讲正文主结论，而是整理：

- Appendix B：局限与未来方向
- Appendix D：模型与训练
- Appendix E：实验设置
- Appendix F：交叉验证
- Appendix G：正式理论框架

它的作用是帮助我在复习时回答两个问题：

1. 这篇论文结论为什么可信
2. 这篇论文还有哪些边界和没解决的问题

---

## 一、Appendix B：局限与未来方向

作者提出了三个非常值得记的方向。

### 1. Beyond Rigid Taxonomies
未来不一定永远只选 joint 或 EE、absolute 或 delta，而可能做：

- hybrid representation
- adaptive action space
- 随任务阶段切换动作表示

例如：

- reaching 阶段用 task-space delta
- fine manipulation / contact 阶段切到 joint-space

### 2. Scaling to High-DoF and Dynamic Tasks
当前结论主要建立在典型机械臂 manipulation 上。

未来还需要验证：

- humanoid
- multi-finger hand
- dynamic task
- deformable manipulation

这些更复杂场景下，当前规律是否还成立。

### 3. Unifying Action Spaces for Generalization and Transfer
作者承认：

- generalized setting 下 task-space 的优势目前很明显
- 但这可能和当前 foundation model 的 pretraining regime 有关

未来如果 joint-space 预训练做得更好，也许会改变这个格局。

---

## 二、Appendix D：模型与训练怎么搭

这部分最值得记的是：

### 1. 模型骨架统一
作者使用：

- FiLM-conditioned ResNet-18
- Transformer encoder-decoder

### 2. 覆盖两类代表性 policy 范式

- Regression-based policy（更像 ACT 风格）
- Flow-Matching-based policy（更像 DP / diffusion 风格）

### 3. 训练配方尽量统一
比如：

- AdamW
- batch size 512
- lr 1e-4
- cosine scheduler

### 我的理解

Appendix D 的价值不在于背参数，而在于说明：

> 作者尽量固定模型框架和训练配方，把主要比较焦点放在 action space 本身。

这使得后面的结论更可信，不太像“谁家模型更复杂谁就赢”。

---

## 三、Appendix E：实验场景为什么有诊断性

Appendix E 说明作者不是随便选任务。

### 真机平台

- single-arm AgileX
- dual-arm AgileX
- AIRBOT

### 真实任务是递进式设计的

1. Touch Cube：测空间精度
2. Pick Cup：测接触与夹爪时序
3. Pick & Place：测长时序误差累积
4. Bimanual Transfer：测双臂协调

### 为什么这个设计很聪明
它在逐步放大 action space 的不同弱点：

- 定位能力
- 接触控制
- temporal drift
- multi-arm coordination

### 额外值得记的一点
作者还采用了 6×6 网格协议，尽量减少“物体位置分布太单一导致模型记死位置”的问题。

---

## 四、Appendix F：交叉验证在加固什么

Appendix F 最关键的作用是：

> 告诉我前面的主结论不是偶然实验现象。

### F.1
在 flow-matching backbone 下再次验证：

- `chunk-wise delta > step-wise delta`

说明这个结论不是只在 regression backbone 下成立。

### F.2
在仿真中再次验证：

- `delta > absolute`
- `joint` 在 scaling 下更有优势

说明这些趋势不是某一台真机的偶然 artifact。

### F.3
在 multi-task unified policy 下再次验证：

- 前面的整体趋势依然 robust

### 我的理解

Appendix F 的意义在于：

> 它把正文里的结论从“看起来合理”进一步推向“跨 setting 复现得比较稳”。

---

## 五、Appendix G：最值得长期记住的 formalization

这一部分是我认为最值钱的附录。

### 1. 动作空间设计可以统一看成两阶段

#### Stage 1：Temporal Decoding
决定的是：

- absolute 还是 delta
- chunk-wise 还是 step-wise

#### Stage 2：Spatial Mapping
决定的是：

- joint-space 直接执行
- task-space 还要经过 IK

### 2. 为什么 step-wise 一定更不稳
作者从结构上证明：

- step-wise 相当于一个累计矩阵
- 噪声放大因子会随 horizon 增长

所以它更差不是碰巧，而是数学结构决定的。

### 3. 为什么 absolute 更稳但更难学
作者也明确承认：

- absolute 对 long horizon 更稳
- 但学习难度高得多

### 4. 为什么空间轴本质是 learnability vs stability 的权衡

- task-space：几何直观，但引入 IK 数值条件问题
- joint-space：数值上更稳，但感知到动作的映射更难学

### 我个人最想记住的一句话

> action parameterization fundamentally governs a core trade-off between learnability and stability

这几乎就是整篇论文的总灵魂。

---

## 六、这篇论文的边界感应该怎么保持

我后面复习时必须提醒自己：

1. 当前结论主要来自典型机械臂 manipulation，不要轻易推广到 humanoid、灵巧手或极动态任务。
2. task-space 在 transfer 里更强，不代表这就是永恒真理，未来也可能被 joint-space 预训练挑战。
3. chunking 和 horizon 目前仍有较强经验成分，理论还不完整。

也就是说：

> 这篇论文已经给出了很强的经验规律和部分理论解释，但还没有把 action space 设计彻底“一劳永逸”地解决掉。
