---
layout: post
title: "cuda graph"
description: ""
---

https://zhuanlan.zhihu.com/p/700224642

cuda graph是什么？

cuda graph就是一张图，里面记录有很多节点。

创建cuda graph的方式有两种：第一种自己手动添加节点组织图，第二种捕抓stream上面的操作来添加节点组成图。

如果是捕获stream上面的操作，其实不会真的执行，只会添加node。

gpu上面的指令大部分都是可以的，tensor.to("gpu")必须是pinned_memory和non_block,gpu到cpu的指令比如tensor.to("cpu")的时候要执行cudaasyncmemcpy。

host的node也可以添加，但是stream不会主动捕获 要手动添加节点，并且host node不能执行cuda api否则会报错。

如果想要捕获多个stream并行kernel，这时候一般是不同stream之间通过event同步可以感染创建graph。

还有一些细节：cuda graph每次执行都会有自己的一个pool。。
