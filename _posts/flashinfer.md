---
layout: post
title: "flashinfer"
description: ""
---

https://arxiv.org/abs/2501.01005
注意，下面讨论的都是decode阶段。

创新点：
1.kv cache统一成block sparse attention

2.composable sparse format：split kv，shared prefix和private suffix分别计算：比如三个q1 q2 q3用相同的shared prefix一个block，
其他的suffix分别block最后合并。

3.Jit custom attention：jit手写算子定义q o transformer

4.shceuling从attention kernel解耦，动态取tile 而不是固定128（提供很多选项）

5.动态拆分请求并负载均衡，32k -> 8+8+8+8K，然后根据不同的block queue的负载决定不同的任务，并且兼容CUDA Graph。大概就是work queue写到cuda graph buffer里面，
每一次运行之前cpu plan()都去buffer里面把work metadata(request、kv chunk、work tile、CTA balance、reduction map)写进去，每一个CTA负责什么work。
gpu负责运行persistent kernel ->  读plan metadata -> 计算attn
cudagraph的地址永远都是0x100000静态，申请很大的地址，然后内容动态。

0.CTA是什么，就是kernel里面的block吗？怎么让一个block动态申请这么多任务的？
1.什么是persistent kernel，为什么要用persistent kernel：（persistent kernel就是持久化kernle，正常kernel执行完一个任务就消失了，
这里的kernel就是生产者消费者，因为cudagraph的griddim固定有这么多block 所以是persistent kernel。graph里面的blockdim、workspace pointer也是固定的）
2.这里动态切分kv长度其实是不是就是dynamic cp？差不多。一个是单gpu上面切， 一个是切了cp分配到不同机器上面。
