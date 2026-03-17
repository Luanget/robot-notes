---
title: RQ2 / RQ3：主结果与趋势
---

# RQ2 / RQ3：主结果与趋势

这一篇负责整理整篇论文最核心的主实验结论。

建议把它分成两条主线来记：

- **时间轴：delta vs absolute**
- **空间轴：joint vs task**

然后再看这些结论在 scaling 和 transfer setting 下会怎样变化。

---

## 一、RQ2：在实现细节调对之后，总体规律是什么

在进入 RQ2 之前，作者已经把 RQ1 的结论用于实验设置：

- delta 统一改成 `chunk-wise`
- delta 使用更短的 execution horizon
- absolute 使用更长的 execution horizon

也就是说，后面的比较是在“尽量公平的实现”下进行的。

---

## 二、时间轴结论：delta 几乎是确定性优势

作者的结论很强：

- 在不同平台、任务配置、模型范式下
- `delta` 都持续且显著优于 `absolute`

### 为什么

作者给出的解释主要有两点：

1. **absolute 太难学**
   - 要从图像直接回归全局目标状态
   - 学习问题更难

2. **delta 更符合现代 IL 的 inductive bias**
   - 把动作预测问题局部化
   - 更像“下一步往哪修一点”
   - 更容易学到稳定模式

### 我的个人理解

在机器人模仿学习里，网络通常更擅长学“局部连续修正”，而不是“一眼看图直接给全局答案”。这也是为什么 delta 的优势几乎贯穿全文所有 setting。

---

## 三、空间轴结论：Joint 通常更强，但不是绝对铁律

总体上，作者观察到：

- `joint-space` 通常优于 `task-space / EE`
- 但这个结论不像 delta 那么“硬”
- 它会受到平台、任务和学习范式影响

### 为什么 Joint 往往更强

- 不依赖 IK
- 控制链条更短
- 执行层更稳
- 一旦模型够强，就能学出 joint-space 的复杂结构

### 为什么这个优势不是无条件的

- joint-space 更难学
- 需要更强模型、更长训练、更大数据
- 在资源有限时，task-space 仍然很有竞争力

---

## 四、为什么 Flow Matching 特别适合 Joint Space

这是整篇论文非常重要的一个观察。

作者发现：

- 在 regression-based backbone 下，joint 相比 EE 的优势还不算特别夸张
- 但在 flow matching / diffusion 风格的生成模型下，joint-space 的优势明显放大

### 我的理解

这是因为 joint-space 的动作分布常常：

- 更复杂
- 更非线性
- 更可能多模态

如果只用简单 MSE 回归：

- 可能会更倾向于“平均答案”
- 对复杂 joint 分布不够友好

而 flow matching 更擅长表示复杂多峰分布，所以更容易释放 joint-space 的潜力。

这个理解也和我自己前面的直觉一致：

> 同一个末端目标可能对应多个合理关节解，joint-space 的问题不是没有规律，而是规律太复杂，不适合被简单回归粗暴平均。

---

## 五、如何读 Table 1

不要背数字，只看三层结构：

### 1. 看 temporal
同样空间下，`delta` 几乎总比 `absolute` 好。

### 2. 看 spatial
在很多标准 setting 下，`joint + delta` 常常是最好或接近最好的组合。

### 3. 看 model paradigm
`joint-space` 在强生成模型（DP / flow matching）下优势最明显。

所以 Table 1 不是在告诉你某个偶然最高分，而是在说明：

> `joint + delta` 在标准单平台 setting 下最容易成为性能上限较高的组合。

---

## 六、RQ3：随着数据和训练增加，结论会变吗

作者继续做 scaling analysis，研究：

- 数据量增加后会怎样
- 训练 epochs 增加后会怎样

### 结论 1：delta 的优势依然非常稳
无论数据量和训练轮数怎么变化，delta 基本都继续压过 absolute。

### 结论 2：Joint 的优势会越来越明显
随着：

- 数据更多
- 训练更久
- 模型更强

joint-space 的优势会被进一步放大，尤其在 regression-based policy 上更明显。

### 我的理解

如果说 task-space 更像“先天容易学”，那 joint-space 更像“后劲更足”。

- 资源少时：task-space 可能已经表现不错
- 资源多时：joint-space 的执行层优势和本体结构优势会越来越被模型学出来

---

## 七、Advanced Regimes：为什么 task-space 会在泛化场景中反超

这一部分是全文最值得玩味的地方。

作者在：

- cross-embodiment setting
- foundation model transfer setting

中发现：

- delta 依然比 absolute 强
- 但空间轴上，`task-space / EE` 会表现出更明显优势，甚至在一些情形下超过 joint-space

### 原因

作者给出的解释是：

- task-space 更 `embodiment-invariant`
- 它抽象掉了具体机器人的关节结构
- 更容易把知识从一个机器人迁移到另一个机器人

### 我的理解

这个结论特别成熟，因为它说明：

- 单平台最优解
- 和跨平台泛化最优解

根本就不是同一个问题。

如果目标是：

- 固定机器人性能最大化 → joint 更优
- 跨机器人知识迁移 → task / EE 更优

这不是前后矛盾，而是优化目标本身变了。

---

## 八、这一篇最终该怎么记

### 最短版本

- 时间轴：`delta > absolute`
- 空间轴：标准 setting 下 `joint` 往往更强；泛化 setting 下 `task / EE` 更强

### 更完整版本

在现代 imitation learning setting 下，properly implemented delta 几乎始终是更好的 temporal abstraction；而 spatial abstraction 则取决于目标：固定平台性能优化更偏向 joint-space，跨本体泛化和 transfer learning 更偏向 task-space。
