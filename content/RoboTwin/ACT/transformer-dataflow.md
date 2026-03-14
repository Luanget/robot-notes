---
title: ACT 中 Transformer 数据流转
---

# ACT 中 Transformer 数据流转

## 1. 这篇笔记要解决的问题

这篇笔记主要回答以下问题：

1. ACT 中 transformer 的输入由哪些部分组成  
2. encoder 如何把输入变成可供查询的 memory  
3. decoder 如何把 query 变成动作隐藏表示  
4. 最终动作是如何输出的  

---

## 2. ACT 中 transformer 的输入组成

ACT 主 transformer 的输入并不是单纯的图像特征，而是由多种 token 共同组成：

### 2.1 latent token
- 来源：CVAE encoder 输出的 latent_input
- 作用：提供动作模式条件

### 2.2 proprio token
- 来源：机器人状态 qpos 的线性投影
- 作用：提供当前机器人本体状态

### 2.3 image tokens
- 来源：多路相机图像经过 backbone 和 input_proj 后的特征
- 作用：提供环境观察信息

### 2.4 query embeddings
- 来源：learned query embeddings
- 作用：在 decoder 中对应未来动作槽位

---

## 3. Encoder 的数据流

### 3.1 输入序列的构造
encoder 输入序列可以表示为：

[latent, proprio, img_1, img_2, ..., img_HW]

其中前两个是额外 token，后面是图像 flatten 后得到的空间 token。

### 3.2 encoder 的核心作用
encoder 通过 self-attention，让每个 token 与所有 token 交互，从而完成多模态信息融合。

其输出不是单个向量，而是一组上下文化后的 token 表示，即 memory。

### 3.3 encoder 输出的含义
encoder 输出的 memory 可以理解为一个“已经融合了图像、状态、latent 条件的信息库”，供 decoder 后续查询。

---

## 4. Decoder 的数据流

### 4.1 decoder 的输入
decoder 输入包括：
- 全零初始化的 tgt
- query embeddings
- encoder 输出的 memory

### 4.2 query 的意义
每个 query 对应未来动作序列中的一个位置。  
由于 query 参数位置固定，且训练时监督位置固定，因此每个 query 会逐渐学成一个稳定的时间槽位。

### 4.3 decoder 的工作过程
decoder 每层主要包括三步：

1. query-query self-attention  
   建立不同动作位置之间的结构关系

2. cross-attention  
   每个 query 从 memory 中提取与自己最相关的信息

3. FFN  
   对得到的隐藏表示做进一步非线性变换

### 4.4 decoder 输出的含义
decoder 最终输出的是每个未来动作位置对应的隐藏表示 hs。

---

## 5. Action Head 输出

decoder 输出 hs 后，通过 action_head 投影到动作空间：

a_hat = action_head(hs)

于是得到整个动作 chunk 的预测结果。

同时也可以通过 is_pad_head 预测 padding 信息。

---

## 6. 我的理解

我认为 ACT 中 transformer 的核心思想是：

1. 先通过 encoder 把图像、机器人状态和 latent 条件融合成统一 memory  
2. 再让 decoder 中的每个 query 去 memory 中提取对应该时刻动作所需的信息  
3. 最后通过线性层把隐藏表示映射为具体动作  

因此，ACT 的 transformer 不是简单地“从图像回归动作”，而是一个“条件化的 memory 查询与动作解码结构”。

---

## 7. 复习时最应该记住的点

1. encoder 的输入不只有图像，还有 latent 和 proprio token  
2. encoder 的输出是 memory，不是动作  
3. decoder 中每个 query 对应一个未来动作槽位  
4. cross-attention 是动作位置从 memory 中取信息的关键步骤  
5. action_head 负责把 hidden state 映射到动作空间
