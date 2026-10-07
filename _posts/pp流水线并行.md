---
layout: post
title: "pp流水线并行"
description: ""
---
https://zhuanlan.zhihu.com/p/685838198

https://zhuanlan.zhihu.com/p/681363624

朱然：
https://mp.weixin.qq.com/s/vCy6ga5EA2dzvFoL8p6QjA


## GPipe
<img width="927" height="294" alt="image" src="https://github.com/user-attachments/assets/e998dada-88a0-40fe-9096-ab4abd3aec8a" />
减小空泡


## 1F1B
<img width="1440" height="815" alt="image" src="https://github.com/user-attachments/assets/f31c0535-936e-4d2d-8708-8cfbcdf2d032" />
相比于GPipe没有减小空泡，但是减小的激活值。

## 1f1b-i（virtual pipeline）
<img width="669" height="302" alt="image" src="https://github.com/user-attachments/assets/430b91dd-f29b-41a4-b5c5-797553e96012" />
减小了v倍空泡，但是通信增加了v倍。

注意，这里有几个细节。
- 这里的形状是梯形
- G(microbatch_group_size_per_vp_stage)=pp，1f1b-I的调度是以这个为当前chunk调度的（f/b队列连续执行当前chunkG个micro-batch）
- warmup数量为N_{warmup}(r) = min(MV, 2(PP-rank-1)+(V-1)G+E)。(V-1)*G就是简单的不考虑当前chunk，PP-rank-1的意思是有多少个rank要传递 乘2是backward是forward的两倍。

## zero-bubble-梯形
<img width="1440" height="189" alt="image" src="https://github.com/user-attachments/assets/379d856d-348f-4bc1-a82a-1554f024704f" />
把dx和dw拆开来了，合理的调度让空泡更小

**这里是怎么调度dx和dw和f的？**

## zero-bubble-平行四边形
<img width="1440" height="251" alt="image" src="https://github.com/user-attachments/assets/809877de-159d-4b83-a9df-9d4e28c20970" />
前面的机器在warmup阶段不空等，而是一直forward保存激活值。但是显存也翻倍了。

## zero-bubble-v
<img width="1440" height="235" alt="image" src="https://github.com/user-attachments/assets/471ea5b3-fa9a-4cf5-885a-0c93ef932e2f" />

**平行四边形显存减少到了梯形级别。**
原理是:
- 之前的layer获得层不平均，现在的layer相对平均。比如gpu0每次切分的layer前向第一个，反向最后一个。显存相对均匀
- 通过拆分调度dx和dw可以提前释放显存


**把1f1b-i换成了-v的调度。（如果-i会不会减少显存占用？）**
会，但是不会减少这么多。

假如有四个stage，pp=2，vpp=2。
gpu0:S0,S3;gpu1:S1,S2

Zero-Bubble的贡献主要是：
- dx和dw的拆分，可以提前传输激活值，减少空泡
- 梯形到平行四边形，减少空泡
- zerobubble-v，v形传输 减少了显存占用

## dual-pipeline
<img width="1716" height="306" alt="image" src="https://github.com/user-attachments/assets/d23579b9-a680-42cf-8c83-097ff658e14a" />
<img width="1720" height="228" alt="image" src="https://github.com/user-attachments/assets/ce57aaff-666c-4372-89ad-94c4ea213984" />

在zero-bubble-v的基础上，把layer数量翻倍了并且规定了方向。
<img width="1446" height="834" alt="image" src="https://github.com/user-attachments/assets/63835cd5-62c3-4055-9bb3-a4b7f51fb7d6" />

然后讲一个batch切成两个mini-batch。根据上述图，进行two-batch-overlap的操作。
<img width="1532" height="1066" alt="image" src="https://github.com/user-attachments/assets/aefa6ff7-b9be-496e-90e9-40a27078154d" />

## 朱然
在1f1b-i的基础上，稳态阶段直接进行two-batch-overlap，改动非常小。
<img width="1080" height="238" alt="image" src="https://github.com/user-attachments/assets/10d236a8-254c-4ff9-bb14-1850fa24ea5b" />
