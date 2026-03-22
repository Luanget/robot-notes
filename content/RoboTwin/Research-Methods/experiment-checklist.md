---
title: 实验设计检查清单
---

# 实验设计检查清单

这个页面是给真正开跑前用的。

如果一个实验跑了很久，最后却发现变量没控制住，通常不是不会训练，而是前置检查不够。

---

## 1. 当前实验要回答什么问题

- 这组实验到底要回答哪个问题？
- 它是在验证方法，还是在排除解释？

## 2. 当前唯一变化的变量是什么

- action space？
- auxiliary loss？
- profile？
- batch size？
- seed？

## 3. 哪些东西必须保持一致

- task
- task config
- data amount
- seed
- train/eval setting
- ckpt 路径
- profile / batch / grad accumulation

## 4. 会不会把工程改动误判成方法效果

- train.sh 是否同步？
- DP 和 DP_geoaux 是否都修了相同底层问题？
- workspace / dataset 的底座修复有没有同时应用？

## 5. 现在最关键应该看的指标是什么

- train / val loss
- aux loss
- rollout 是否开始有信号
- 成功率
- 是否出现 train 好但 eval 差的情况

## 6. 如果结果不如预期，优先检查哪一层

按顺序：

1. 实现层
2. 优化层
3. 数据层
4. 表示层
5. 方法本身

---

## 一句话总结

实验开始前真正该问的不是“能不能跑”，而是：

> **这个实验跑完之后，我到底能排除什么，确认什么。**
