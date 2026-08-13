---
layout: post
title: "AWQ、SmoothQuant、GPTQ"
description: ""
---

https://chatgpt.com/share/6a7c204e-b590-83ec-9143-8c25f307a643：动态量化比静态量化效果差多，所以一般用静态量化lieanr而不是Q @ K^T这种激活值量化。其次即使保证效果也没太大收益，可以看上面的gpt。（hopper的tc / cuda大概是16:1,blackwell的是32:1，并且因为hopper的wgmma没有tmem和umma会抢register，所以其实也是32:1。[128,128],[128,128]的矩阵乘法，blackwell的tensorcore要1024个cycles，cuda core的exp单元也要1024cycles刚好一样，所以bf16是甜点）；
纯激活值量化可以KV量化，在线反量化回来计算。

logits用fp32、attn_score、router用fp16，其他的是fp8.

# RTN量化
就是 q = round(x / delta), x' = delta * q

# AWQ：仅仅量化w，用激活值统计分布处理w的缩放强度。
https://arxiv.org/abs/2306.00978
https://chatgpt.com/share/6a79c37f-745c-83ec-a2c4-63ce609ea665
这篇gpt讲的很透彻。

大概意思就是这篇文章发现int4直接量化误差很大，但是保护0.1～1%的权重之后就能提升很多精度。

具体是哪1%的权重呢？作者是这样发现的：
1.按照weight channel的平均值权重来看，跟平均值没什么关系
2.主要是跟激活值有很大的关系， y = x * （w + delta_w）
所以作者这篇论文叫做activation-aware weight quant...

在输入给attn之前的x，把x的channel维度做平均值a_j，得到偏差比较大的维度，论文认为这一channel比较危险。
然后给把让

$s_j = a_j^t,0<=t<=1$, 
 
让weight这一channel乘上 $s_j$

这样被int round损失就没有这么大了。

但是乘了scale必须除以scale才能保证数学结果等价。能不能像矩阵吸收一样，不多算一次矩阵乘法次呢？

论文是在前一层的输出channel除以scale了。比如attn的q前面的RMSnorm。但是Gelu不是线性的，不能线性映射 只能在线除以scale了。

```
@triton.jit
def int8_gemm_tiled_kernel(
    # 指针
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,  # 每张量缩放因子（FP32）
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 元参数：分块大小
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 线程块在输出矩阵中的位置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 该块负责的行的范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器：FP32！直接以FP32累加，避免后续类型转换开销
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 遍历 K 维度，每次处理 BLOCK_K 个元素
    for k in range(0, K, BLOCK_K):
        # ---- 1. 创建 A 的分块指针 (INT8) ----
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        # ---- 2. 创建 B 的分块指针 (INT8) ----
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

        # 加载 INT8 数据，自动提升为 INT32 供 tl.dot 使用
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0)
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0)

        # ---- 3. 核心矩阵乘：INT8 * INT8 -> INT32 ----
        # tl.dot 要求输入至少是 INT16，这里 INT8 会自动扩展，输出 INT32
        acc += tl.dot(a, b).to(tl.float32)  # 关键：INT32 转 FP32 后累加

    # ---- 4. 量化反量化：应用缩放因子，得到最终 FP32 输出 ----
    scale_a = tl.load(scale_a_ptr)  # per-tensor 激活缩放
    scale_b = tl.load(scale_b_ptr)  # per-tensor 权重缩放
    c = acc * (scale_a * scale_b)   # 公式：C_fp32 = (A_int8 * B_int8) * (scale_a * scale_b)

    # ---- 5. 存储 FP32 结果 ----
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```
这里有一段triton量化代码，细节就是：先求出block负责的offsets。不管怎么样，都要先求出block的对应区域的ptr（这里要利用triton的广播机制），
最后在load、store的时候也要广播和mask一下。求block ptr用的是+的广播，load store用的是&的广播。



# SmothQuant：激活值和权重同时量化，将激活值量化难度迁移到权重上
