---
layout: post
title: "GRPO"
description: ""
---

https://zhuanlan.zhihu.com/p/1888311680880080185

# GRPO公式

<img width="1374" height="488" alt="image" src="https://github.com/user-attachments/assets/0405eb28-f198-42a4-9614-89cddef4b92f" />

也就是sequence-level：一个group内，先每一个sequence求梯度，sequence内部Loss加起来求平均数，然后再把每一个sequence梯度加起来求平均

**为什么sequence-level更容易让回答变长？**

考虑四种情况：正确而短的回答、正确而长的回答、错误而短的回答、错误而长的回答。正确而短的回答，token权重大，长的回答token权重小。

错误而短的回答权重也大，训练后token出现概率很小；错误而长的回答权重小，训练后token出现概率相对大。

所以在正确性大的情况倾向于短回答，正确性小的情况下倾向于长回答。

```
response 1
token loss 求平均 ─┐
                   │
response 2         ├→ response 再平均
token loss 求平均 ─┘
```

# DAPO
<img width="1368" height="262" alt="image" src="https://github.com/user-attachments/assets/2d59fc13-776e-4b1f-9ede-c5bd68b2bdb5" />

- token-level：所有token的loss加起来求平均数。
- 上界clip，鼓励优秀的探索
- 动态采样，全队全错的样本丢弃
- 长度惩罚：超出上下文逐步降低reward，完全超出就设置为0
- 删除KL，认为LongCoT模型应该偏离原模型






# DSPO
首先rollout会拆成mini-batch，所以importance ratio(logp)是防止off-policy走太偏。

作者认为grpo的reward是sequence-level的，所以sequence同一个reward，但是importance ratio和clip ration却是token维度的，非常不自然。

<img width="1382" height="276" alt="image" src="https://github.com/user-attachments/assets/30d03385-2b85-4fee-8f29-fdff0a0aadb6" />

这里用了sequence-level的clip和ratio，具体公式是<img width="1282" height="226" alt="image" src="https://github.com/user-attachments/assets/8fcd2460-3136-4c6b-9a13-e447677886bd" />

GSPO可能会clip 15%的token，grpo只会clip 0.13%token，认为grpo有太多noisy token gradient。

在moe阶段，grpo需要r3，但是gspo不需要，因为只关系整条sequence的gridient，不关心几个token的router变化，也可以正常收敛

