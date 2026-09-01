---
layout: post
title: "FA"
description: ""
---
https://chatgpt.com/share/6a7d6b03-5bb4-83ec-93b4-6134717e5dc1

# FA1
[fa1:](https://arxiv.org/abs/2205.14135)
主要讲了Attn不仅仅是计算瓶颈，而且也是IO瓶颈。因为Q @ K ^ T的这里有n^2 HBM读写。
主要是onlinesoftmax让在smem上，优化了IO瓶颈，少访问HBM。

【bs，head，num_split】的kernel，每一个CTA的逻辑是这样的
```
for kv in block_N:
  for q in block_M:
      算S[m,n]
      更新softmax
      更新O[m]
这里一个CTA负责一个完整的【bs，head】，但是第二个for循环要lse merge，reduction，不过是CTA内部。
```
<img width="1748" height="778" alt="image" src="https://github.com/user-attachments/assets/629600ee-15d3-4493-8c33-a5f61fb6d158" />

BLOCK内部的warp，是同一个KV 不同的Q，所以warp同步__syncthreads之后就可以可以lse merge同一个O了。

# FA2
[fa2:]https://arxiv.org/abs/2205.14135

【bs,head,q_seq】
```
for q in block_M:
  for kv in block_N:
    算S[m,n]
    
```
<img width="1392" height="844" alt="image" src="https://github.com/user-attachments/assets/eb2ee775-9626-45f3-91a7-25dd4482304b" />
fa2的block切分是直接切Q，并且warp之间也不用reductin同步了。

IO降下来之后，新的瓶颈在于SM利用率、CTA/warp任务、shared memory通信
更多SM同时干活，让warp少通信。

怎么发现的？gemm可以到理论峰值80-90%，但是flash forward只有30-50%，backward只有25-35%。IO优化的很好，但是Tensor Core吃不满。

问题一：non-matmul FLOPs太贵，exp sum scale...（这也是attn很少量化的原因）hopper差了16倍，blackwell差了32倍。
想办法把时间花在Tensor Core GEMM上面

问题二：fa1只有（batch，head）并行，小batch喂不满SM。长seq 并行的CTA不够多

问题三：warp之间存在不必要的通信：一个warp里面共享q不同kv,要reduce一下，才能得到O。

S = Q @ K ^ T , P = softmax(S)

问题一解法：、维护O'是没有除以softmax的l，只有最后才除以softmax的l。这样可以减少division和rescaling。

不保存m，l,只保存logsumexp

问题二解法：seq parall，提高CTA数量

```
FA1：

parallel dimensions=batch×heads

FA2 变成：

batch×heads×Q_blocks
```	​
就是说一个CTA负责一个Q（其实也就是token），这样就不需要同步了。（类似于ulyssess和lss transform的区别）

对于一个CTA内部有很多warp，相比于fa1，fa2也是不同warp负责不同的q保持独立。

# FA3:
https://arxiv.org/abs/2407.08608
问题：FA2只有35%的H100利用率，但是GEMM可以做到80%-90%。

怎么发现的呢？H100有若干可以同时工作的独立硬件单元

Tensor Core 上 FP16 的算力为 4096 FLOPs/cycle，MUFU 的 FP16 算力仅为 16 OPS/cycle（计算exp的硬件）。在128*128的情况下，gemm的时间是softmax的两倍。
```
TMA + Tensor Core + Cuda/SFU
```

https://zhuanlan.zhihu.com/p/17533058076
https://chatgpt.com/share/6a7f21ae-5eb8-83ec-bd4d-bdc9dc334e9b

主要是三个优化：
- wg specialzation：生产者 消费者（搬运q，k_j,v_j+1）
- wg之间的overlap，wg内部的overlap(wg之间：想办法让mma和softmax同时利用：两个mma异步指令开始后，softmax之前signal一下。
- wg内部：想办法减少串行等待：q@k_j,k^j+1@v_j+1)
```
                iteration j         iteration j+1

Tensor Core:    Q K_{j}^T      ||     P_{j+1}V_{j+1}
                                      ↓wait_wgmma<1> (等待只有一个mma)
CUDA/SFU:                          softmax_{j+1}
```

这里其实可以有更多的stage overlap（比如存着两个PV tile）

- attn量化
Hadamard降低离散值
v的tranpose



