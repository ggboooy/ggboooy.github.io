---
layout: post
title: "moe"
description: ""
---

# pd分离怎么回答？
- prefill和deocde性质不一样，prefill是computed bound 通信量大，decode是memory bound，通信量小。从算数强度可以看出来
- 两者采取的优化手段不一样：prefill可以开一些cp和pp缓解计算压力，decode一般是大ep+投机解码，这样memory bound压力小并且转化为compute bound，算数强度大了。
- prefill一般deepep采取norm模式，内存排布、通信方式、grouped_gemm怎么算的、如果训练可以paged-stashed+cuda graph。
- decode一般用low-latency模式，内存排布、通信方式、grouped_gemm怎么算的、
- 注意，两者的two-batch-overlap也不一样，这里decode通信用的sm少所以这样搞。
