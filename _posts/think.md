---
layout: post
title: "think"
description: ""
---
# moe tuning
moe在不同M（输入的token数量）的时候会针对性调优。
<img width="2634" height="224" alt="image" src="https://github.com/user-attachments/assets/7eab0499-eaed-491f-8583-060879f78305" />
sglang会告警上述。
<img width="2472" height="776" alt="image" src="https://github.com/user-attachments/assets/9ee805bf-6ed5-4210-84b3-915e3a41d6db" />

看耗时moe还是挺高的。

moe tuning主要是针对如下算子进行调优：
```
  {
      "BLOCK_SIZE_M": ,   # token 维度分块
      "BLOCK_SIZE_N": ,   # 输出维度分块  
      "BLOCK_SIZE_K": ,   # 内积维度分块
      "GROUP_SIZE_M": ,   # L2 亲和 swizzle
      "num_warps": ,      # 每 block warp 数
      "num_stages": ,     # 软件流水深度
  }

"32": {
          "BLOCK_SIZE_M": 16,
          "BLOCK_SIZE_N": 128,
          "BLOCK_SIZE_K": 128,
          "GROUP_SIZE_M": 32,
          "num_warps": 4,
          "num_stages": 3
        }
比如这个，指的是一个block内部有4个warp，按照路由的token以16为一组block（如果小于就padding）
按照
  acc = 0
  for k in range(0, 2048, 128):
      acc += A[:, k:k+128] @ W[k:k+128, :]
这样来加法。

GROUP_SIZE_M
```

cuda内存事务
```
SM load一次的顺序:register->smem->L1->L2->gmem，然后再逐级回填。

https://zhuanlan.zhihu.com/p/577412348
gmem的读取单位是事务，每次都连续读32byte连续内存单位（也称为sector）。
gmem主要讲究合并访存，希望读取的东西都是内存连续的。也有float2 float4让一个线程读连续的float避免内存事务浪费。

smem的读取单位是bank，每次都读到某一个bank上面也有swizzle来优化smem。

L2 cache：GROUP_SIZE_M？
```
这6个参数没有全局最优值，最优组合依赖于：
- GPU型号（SM数、L2大小、TensorCore shape），
- 算子 shape (E, N, K, topk)：E是expert总数、N是moe降维的维度（gate+up），K是expert的hidden size
- 数据类型（fp8/bf16/int8）
- 量化布局（per-tensor / per-channel / block-wise）。
