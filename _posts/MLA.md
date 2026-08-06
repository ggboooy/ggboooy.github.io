---
layout: post
title: "MLA"
description: "先粗看一遍，后面还会仔细学习"
---

# MLA
https://zhuanlan.zhihu.com/p/19585986234
<img width="3026" height="2894" alt="image" src="https://github.com/user-attachments/assets/1281f464-49c5-46eb-8751-8438f9892ca8" />
注意，q的up_proj之后才得到的RoPE。而kv down_proj之后就得到了K的RoPE。

1.MLA怎么应用RoPE
首先 RoPE会的角度跟token dimension（比如token的embedding维度是1000，位置1和位置50是不一样的）、token position（seq_len）有关系
<img width="1636" height="1028" alt="image" src="https://github.com/user-attachments/assets/bc177930-cf5f-47fc-9e75-6a9a5800d89f" />
正常的MHA W矩阵乘一次，缓存的KV就自带RoPE了，后面可以重新复用。
MLA缓存的是低秩c_{kv,t}，这时候没有自带RoPE，如果每一次都升秩得到KV，还要在重新应用RoPE 成本就太高了。
MLA的处理是把KV的RoPE拆开，单独拆成一段小维度RoPE单独缓存。（low_rank维度是512，RoPE维度是64），然后给每一个head都concat上去。

2.MLA的cp：all-gather（lss transformer）、ulyssess


3.矩阵吸收的计算量分析
