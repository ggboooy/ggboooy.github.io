---
layout: post
title: "moe"
description: ""
---

https://arxiv.org/abs/2603.07685

# 确定性算子
- moe expert的累加顺序
- 残差相加是先bf16还是先累加
- deep_gemm，batch_invarient
- DSA index的replay
- router replay

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

norm模式这里其实必须要所有gpu等待传送要接受的数量，然后cpu分配显存再dispatch permute。

最新的norm模式技术其实有 paged-stashing。
## 通信
norm需要相同索引的gpu中转 然后再发送，并且收到的token数量是动态的，需要cpu统计数量

low-latency不需要统计数量，直接收所有token

## 显存墙
- recompute
- cp/sp
- offload、zero3

## 通信墙
- deepep
- pp流水线

## 计算墙
- grouped gemm(多流stream启动kernrl、持久化kernerl、FC1 swiglu FC2算子融合)
- cudagraph（device launch、paged-stashing、echo）。注意这里的device launch可以让device选择最优的launch配置，不需要cpu的eager模式？？
- cudagraph开销主要是：python->框架->kernerl launch
- 算子融合（process&permute&unpermute、router&aux）

## 打破三堵墙共同的办法：量化
- padding
- 选择性量化
