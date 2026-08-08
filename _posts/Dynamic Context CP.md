---
layout: post
title: "Dynamic Context CP"
description: ""
---
https://chatgpt.com/share/6a76c026-4fc4-83ec-99a1-84250581c526

主要是为了解决：大EP前提下，attn dp负载不均导致的空等，然后一起进入moe阶段。
varlen情况下的负载均衡策略解决。
https://github.com/vllm-project/vllm/issues/29295
## 1. 先把 Dynamic CP 拆成两层
它实际上有两层：
Dynamic CP
│
├── 控制面：这个 request 用 CP=1 / 2 / 4 / 8？
│
└── 数据面：选定 CP=N 后，Attention 怎么算？
      ├── Decode：sharded KV + partial attention merge
      └── Prefill：
           ├── KV AllGather + local Q
           └── Ring pass-KV / pass-Q

vLLM 的 Dynamic CP RFC #29295 的核心创新主要在第一层，对于不同的请求用不同的CP：
一个长请求可以临时占用多个原本独立的 DP-Attention replica。
例如原来：
DP0: request A
DP1: request B
DP2: request C
DP3: request D
现在可能变成：
短请求 A：CP=1 → DP0
长请求 B：CP=2
            ├── DP1
            └── DP2
超长请求 C：CP=4
            ├── DP0
            ├── DP1
            ├── DP2
            └── DP3
RFC 提出提前建立多个通信 group，根据请求长度选择 CP degree，避免运行时反复创建 NCCL group；scheduler 上再增加 Domain 和 CrossDPScheduler，统一管理多个 DP replica。

## 2. Decode 阶段是DCP
https://zhuanlan.zhihu.com/p/2020086868914499979
<img width="2094" height="738" alt="image" src="https://github.com/user-attachments/assets/ca38b027-25d2-4a77-b08a-9af2c90506e1" />
假设：

context = 8 tokens
CP = 2

KV cache 可以分成：

GPU0:
K0 V0
K2 V2
K4 V4
K6 V6

GPU1:
K1 V1
K3 V3
K5 V5
K7 V7

vLLM 已合入的 DCP 正是这种 interleaved KV layout：token i 存在 i % dcp_world_size 对应的 rank。

现在生成 token 8。

只有一个：

Q8

两个 rank 分别做：

GPU0:
Q8 × [K0,K2,K4,K6]
        ↓
(O0, LSE0)

GPU1:
Q8 × [K1,K3,K5,K7]
        ↓
(O1, LSE1)

最后精确 merge：

(O0,LSE0) + (O1,LSE1)
          ↓
          O8

注意，这里的merge是online softmax的merge。
不仅如此，dcp其实可以在GQA/MLA阶段省显存（部分head头重复存储，从token维度就少存储了）但是现阶段vllm/sglang的dcp强制以来于TP

## 3. prefill阶段是all-gather cp
这里的all-gather cp指的是kv all-gather，q不all-gather（或者x输入 all-gather一次，重复计算一下，q部分丢弃）
这里可以用zig-zag均衡计算。
最后得到各个完整token的o。主要，和原版lss transformer不一样，这里的cp 不一样可能要all-gather一下。


## 4.Dynamic CP 与普通 Static CP 最大区别

普通：

--cp-size 4

所有 request：

4K   → CP4
8K   → CP4
512K → CP4
1M   → CP4

问题是短 request：

Attention kernel = 10 us
通信 = 20 us

CP 反而更慢

Dynamic：

4K     → CP1
32K    → CP1/2
128K   → CP2
512K   → CP4
1M     → CP8

所以本质是：

CP(r)=f(context_len,new_tokens,batch,KV occupancy,network,GPU load)

现实里不能只看 sequence length。

更好的 cost model 应该看：

Prefill:
new_tokens
total_context
KV/Q head ratio

Decode:
KV length
batch size

Global:
各 DP attention load
各 GPU KV occupancy
EP batch balance
通信拓扑
