---
layout: post
title: "AWQ量化"
description: ""
---

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
