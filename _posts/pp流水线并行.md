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

## zero-bubble-梯形
<img width="1440" height="189" alt="image" src="https://github.com/user-attachments/assets/379d856d-348f-4bc1-a82a-1554f024704f" />
把dx和dw拆开来了，合理的调度让空泡更小

**这里是怎么调度dx和dw和f的？**

## zero-bubble-平行四边形
<img width="1440" height="251" alt="image" src="https://github.com/user-attachments/assets/809877de-159d-4b83-a9df-9d4e28c20970" />
前面的机器在warmup阶段不空等，而是一直forward保存激活值。但是显存也翻倍了。

## zero-bubble-v
<img width="1440" height="235" alt="image" src="https://github.com/user-attachments/assets/471ea5b3-fa9a-4cf5-885a-0c93ef932e2f" />
平行四边形显存减少到了梯形级别。

**把1f1b-i换成了-v的调度。（如果-i会不会减少显存占用？）**

device1不存储0和p,2p,3p,vp等这些layer，而是倒着存：0,layer-1... 

