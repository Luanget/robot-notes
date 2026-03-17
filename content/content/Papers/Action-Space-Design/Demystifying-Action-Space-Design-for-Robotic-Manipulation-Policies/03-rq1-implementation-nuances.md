---
title: RQ1：实现细节为什么是决定性的
---

# RQ1：实现细节为什么是决定性的

这一部分的核心思想是：

> 在比较“哪种 action space 更好”之前，必须先把实现细节理清；否则很多表面上的结论其实是不公平的。

作者在这一部分主要研究两个问题：

1. `step-wise delta` 和 `chunk-wise delta` 差在哪里
2. action chunking 的 horizon `k` 能不能单独调

---

## 1. step-wise vs chunk-wise：为什么要分这么细

很多人会以为“delta 就是相对动作”，但作者指出：

> 一旦进入 action chunking 设定，delta 其实有两种完全不同的实现。

### Step-wise Delta
第 `k` 步动作依赖前面所有增量的累加。

### Chunk-wise Delta
整个 chunk 中的每一步都相对于 chunk 起点来定义。

看起来都叫 delta，但误差传播结构完全不同。

---

## 2. 论文在真实机器人上观察到什么

作者在单臂平台三个基础任务上比较发现：

- `chunk-wise delta` 持续优于 `step-wise delta`
- 平均差距可达 10% 以上

这不是轻微改进，而是很显著的性能差异。

---

## 3. 为什么 Step-wise 更差：核心直觉

### Step-wise 的本质

每一步都建立在前一步的预测结果上：

- 第一步错一点
- 第二步沿着前面的错误继续往前走
- 第三步再在“已经偏了”的基础上继续积累

所以误差会像滚雪球一样扩大。

### Chunk-wise 的本质

整个 chunk 内的每一步都相对同一个起点来定义。

这意味着：

- 每一步不会依赖前一步的错误结果
- 误差不容易在时间上串联放大

---

## 4. Appendix G 对这部分的 formalization

作者把两种形式正式写成：

- chunk-wise：`a_{t+k} = s_ref + z_{t,k}`
- step-wise：`a_{t+k} = s_ref + sum(z_{t,1:k})`

接着说明：

- step-wise 对应的是一个下三角累加矩阵
- 其误差放大因子会随 `k` 近似线性增长
- chunk-wise 和 absolute 的误差上界则保持常数量级

所以 step-wise 更差，不是“实验碰巧如此”，而是**结构上就更不稳定**。

---

## 5. horizon `k` 为什么不能孤立调

作者继续研究 execution horizon，发现：

- **delta** 更适合较短 horizon
- **absolute** 更适合较长 horizon

### 我自己的理解

这个结论特别像“局部修正”和“全局目标”的自然差别：

- delta 更像连续小修正，适合短窗口快速纠偏
- absolute 更像直接给全局目标，适合长窗口维持方向一致性

所以：

> `k` 不是一个可以随手抄默认值的常数，而是和 temporal abstraction 强耦合的设计变量。

---

## 6. 为什么 absolute 不是“天然更差”

这部分很重要。

论文并没有说 absolute 没道理。相反，作者承认：

- absolute 在统计上对长 horizon 更稳
- 因为它自带全局锚定，不像 delta 那样参考状态会逐渐过时

但 absolute 的问题是：

- 学习难度显著更高
- 需要从视觉中直接推断全局目标状态

因此在现代 imitation learning setting 下，delta 往往更容易成为更好的实际选择。

---

## 7. 我对 RQ1 的最终理解

RQ1 真正做的事情不是“顺手调参”，而是：

- 先把 delta 的正确实现方式讲清楚
- 再说明 horizon 必须和动作抽象一起设计
- 从而为后面 RQ2 / RQ3 的总体比较打下公平基础

也就是说，RQ1 的作用是：

> 先把“比较规则”理顺，再讨论谁更好。

---

## 8. 这一篇最后该记住什么

### 一句话版本

`chunk-wise delta` 明显优于 `step-wise delta`，而且最优 horizon 依赖动作抽象，不能单独调。

### 更完整版本

现代机器人策略里的动作空间设计不能脱离 action chunking 来看；一旦进入 chunking，delta 的定义会影响误差传播结构，而 horizon 的选择则必须和 temporal abstraction 联合设计。
