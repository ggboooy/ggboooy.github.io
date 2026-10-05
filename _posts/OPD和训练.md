---
layout: post
title: "OPD和训练范式"
description: ""
---

Q：
1.实际的OPD工程实践
2.dpskv4的论文其他内容
3.序列级的。为什么说RL监督信号是稀疏的。。


参考链接：
别再只知道GRPO了！！2026你必须要了解OPD！！！ - JustBeClaw的文章 - 知乎
https://zhuanlan.zhihu.com/p/2019475304599536345

为什么现在普遍基于RL去提升模型的推理能力，而不是通过构造数据做SFT的方式呢？ - Soulflare的回答 - 知乎
https://www.zhihu.com/question/1989082556772135720/answer/1997457198754850615

从技术报告看On-Policy Distillation的崛起: 大模型后训练新范式 - 潜龙勿用的文章 - 知乎
https://zhuanlan.zhihu.com/p/2031101471563962191

On-Policy Distillation (OPD)：起源、发展路线与当今现状 - 翞翞翞的文章 - 知乎
https://zhuanlan.zhihu.com/p/2037285722151989443

# 训练环节
一般来说训练有这几个训练环节：PT（预训练）、SFT（有监督微调）、RLHF（人类偏好的强化学习）、OPD（在线蒸馏）。

首先KL一般有两种：Forward KL和Reserve KL。前者和SFT方向一致,为off-policy覆盖模式（背书）；后者和OPD方向一致，为on-policy寻找模式（补习班）。

```
S:Student_prob,T:Teacher_prob

Forward KL:
T * log(T/S)

Reserve KL:
S * log(S/T)

其中Forward KL可以变形成:
T * log(T) + Loss_SFT
```

从上述公式我们分析：

当S -> 0 也就是学习对于这种没有分布倾向时，
- forward KL->∞,这时候有巨大的惩罚。
- reserve KL->0，这时候有没有什么惩罚。（但是如果T->0 没有得到老师的支持就会KL->∞）。

从公式中我们可以发现，SFT就是背书背答案，模仿老师的一切行为。OPD更加倾向于学生自己探索寻找，要得到老师支持。SFT有曝光误差的问题：推理如果走了训练不一样的路径，就会出现问题。

在后训练实践训练中，一般是SFT，学生通过COT和非COT数据学习基础的解决问题流程，然后再通过RL来做强化学习，强化推理能力。
在解迷宫问题的时候（知道起点终点怎么出去），SFT可认为是背下来了迷宫的路径，而RL是通过之前背下来的迷宫路径自己搜索策略、出路。

OPD我们一般认为有两种作用，
- 解决灾难性遗忘（GLM-5论文中:pt->sft->reasoning rl->agentic rl->general rl->opd)，用opd蒸馏倒数两个流程的数据 避免灾难性遗忘
- 整合数据。(dpskv4:pt->不同领域sft+rl专家->opd)把不同领域的专家数据收敛到pt基模里面。
- 最开始的qwen3，替代RL,节省十倍计算资源。（比如grpo组内要有对比才可以训练，ppo要多一个critic model）。

在实际使用OPD的过程中，只需要把GRPO中的group size:4 -> 1,global batch size:32->1024并且把reward的公式改成teacher的logits就行了。
注意：OPD有两种模式：token-level（teahcer只输出每一个token的logits），vocab-level(teacher告诉你其他的正确答案)。
