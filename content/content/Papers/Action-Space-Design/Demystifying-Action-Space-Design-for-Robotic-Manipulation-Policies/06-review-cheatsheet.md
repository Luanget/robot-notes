---
title: 速查复习页
---

# 速查复习页

这一页只做快速回顾。

适合：

- 一段时间没看论文后快速找回主线
- 做汇报前快速翻一遍
- 记不清某个结论对应哪一部分时临时查阅

---

## 一、这篇论文到底研究什么

研究：

> 机器人模仿学习中，action space 应该怎么设计。

作者把问题拆成：

- spatial abstraction
- temporal abstraction
- action chunking

---

## 二、整篇论文一句话总结

> proper implementation 下，delta 几乎总是更好的 temporal abstraction；而 spatial abstraction 的优劣取决于目标：单平台性能更偏 joint-space，跨平台泛化更偏 task-space。

---

## 三、三个研究问题分别做了什么

### RQ1
研究实现细节：

- chunk-wise vs step-wise
- horizon `k` 和 abstraction 的关系

### RQ2
研究总体趋势：

- delta vs absolute
- joint vs task

### RQ3
研究这些规律在 scaling / transfer / cross-embodiment 下还稳不稳。

---

## 四、最该记住的 6 个结论

1. `chunk-wise delta > step-wise delta`
2. `horizon k` 不能独立调
3. `delta > absolute`（在现代 IL setting 下非常稳）
4. 标准 setting 下 `joint` 往往比 `task / EE` 更强
5. `flow matching / diffusion` 更容易发挥 `joint-space` 的优势
6. generalized setting 下 `task / EE` 会反超 `joint`

---

## 五、我自己的理解版结论

### 1. 为什么 delta 强
因为它把学习问题局部化了，更像“下一步怎么修一点”，对现代模仿学习更友好。

### 2. 为什么 joint 在标准 setting 下强
因为它绕开了 IK，执行更稳；一旦模型够强，就能学出更复杂的 joint-space 结构。

### 3. 为什么 task 在泛化 setting 下强
因为它更 embodiment-invariant，更容易把“手去哪里”的几何语义迁移到别的机器人。

---

## 六、最容易忘的术语

- action space：动作空间
- spatial abstraction：空间抽象
- temporal abstraction：时间抽象
- action chunking：动作分块
- absolute：绝对动作
- delta：增量动作
- chunk-wise delta：块式增量
- step-wise delta：逐步增量
- configuration manifold：构型流形
- embodiment-invariant：本体无关

---

## 七、最值得记住的图和表

- Figure 2：taxonomy
- Figure 3：RQ1（chunk-wise / step-wise + horizon）
- Table 1：总体结果
- Figure 5：scaling
- Figure 6：advanced regimes / transfer

---

## 八、三条最终截图分别对应什么

### 截图 1
**horizon `k` 不能孤立调**

对应：
- RQ1
- Figure 3(b)
- Appendix G 中对 temporal decoding 的解释

### 截图 2
**标准 setting 下，joint + chunk-wise delta 最稳**

对应：
- RQ2 主结果
- Table 1
- RQ3 scaling 对 joint 优势的进一步加固

### 截图 3
**generalized setting 下，task-space 更优**

对应：
- RQ3 advanced regimes
- Figure 6
- Appendix B 中关于 transfer / generalization 的未来方向

---

## 九、这篇论文最成熟的地方

不是给出一个全局唯一答案，而是给出：

- 单平台最优时该怎么选
- 泛化最优时该怎么选
- chunking / horizon 应该怎样配套设计

也就是：

> 它给的是条件化设计原则，而不是口号。

---

## 十、后面如果要继续复习，优先看什么

### 第一次回忆
1. [[Papers/Action-Space-Design/Demystifying-Action-Space-Design-for-Robotic-Manipulation-Policies/01-paper-overview|论文总览]]
2. [[Papers/Action-Space-Design/Demystifying-Action-Space-Design-for-Robotic-Manipulation-Policies/02-taxonomy-and-core-ideas|Taxonomy 与核心概念]]
3. [[Papers/Action-Space-Design/Demystifying-Action-Space-Design-for-Robotic-Manipulation-Policies/04-rq2-rq3-main-results|主结果与趋势]]

### 做展示前
1. 这页速查
2. [[Papers/Action-Space-Design/Demystifying-Action-Space-Design-for-Robotic-Manipulation-Policies/03-rq1-implementation-nuances|RQ1]]
3. [[Papers/Action-Space-Design/Demystifying-Action-Space-Design-for-Robotic-Manipulation-Policies/04-rq2-rq3-main-results|RQ2 / RQ3]]
4. [[Papers/Action-Space-Design/Demystifying-Action-Space-Design-for-Robotic-Manipulation-Policies/05-appendices-and-limitations|附录与局限]]
