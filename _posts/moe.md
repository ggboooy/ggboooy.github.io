---
layout: post
title: "moe"
description: ""
---

https://arxiv.org/abs/2603.07685

还没看完这篇文章，先讲讲自己注意的点。

小M优化：1.swap ab 2.group-M 3.split-K 4.fuse（dispatcher（permute、offset）） grouped gemm

deepep的细节，怎么实现low-latency、norm，通算融合的？

## permute，grouped gemm
dispatch包含了permute，permute是目标expert的传输布局：
```

   张量        Normal 常用的 contiguous layout    Low-latency 常用的 masked layout
  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   输入        [所有 expert 输入段的总行数, K]    [E, C, K]
  ──────────  ─────────────────────────────────  ──────────────────────────────────
   权重        [E, N, K]                          [E, N, K]
  ──────────  ─────────────────────────────────  ──────────────────────────────────
   输出        [所有 expert 输出段的总行数, N]    [E, C, N]
  ──────────  ─────────────────────────────────  ──────────────────────────────────
   分组信息    expert 标记或段边界                masked_m[e]
  ──────────  ─────────────────────────────────  ──────────────────────────────────
   含义        各 expert 的输入段拼在一起         每个 expert 占一个固定容量区域

```
c指的是这个expert的数

## 通信
norm需要相同索引的gpu中转 然后再发送，并且收到的token数量是动态的，需要cpu统计数量

low-latency不需要统计数量，直接收所有token
