---
layout: post
title: "moe"
description: ""
---

https://arxiv.org/abs/2603.07685

还没看完这篇文章，先讲讲自己注意的点。

小M优化：1.swap ab 2.group-M 3.split-K 4.fuse（dispatcher（permute、offset）） grouped gemm

deepep的细节，怎么实现low-latency、norm，通算融合的？

