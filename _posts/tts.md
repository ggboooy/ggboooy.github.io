---
layout: post
title: "tts论文学习"
description: "sglang-omni"
---
# VALL-E
难点是zero-shot-man：训练集没有的一个人的声音，很难能够泛化出他的声音。
VALL-E的目标是模拟出一个人声音，读特定文字。

VALL-E是怎么把audio量化成token的呢？
在一段10*24KHZ的音频里面，用EnCodec压缩成10*75HZ。
从240K的samples -> 750 frame
然后对每一个frame量化成8个独立的codebook。
每个 codebook：都有1024 个 code。
所以一段 10 秒语音最终表示为：
750×8个token
可以想象：
```
时间     RVQ1 RVQ2 RVQ3 ... RVQ8

t=0       24   781   92 ...  314
t=1      625    37  442 ...  123
t=2       98   832  152 ...  517
...
t=749
```
每一个数字都类似 LLM 中的：
token_id
只是词表不是 50K/100K，而是每个 codebook 有 1024 个 token。
codebook具体含义：前面的第一个codebook决定说话内容，第二个codebook决定说话响度，第三个决定音色...

（这里其实类似于搜广推里面的对item量化成token，因为item有1e10太多，不可能都创造一个token，词表不可能这么大，所以想办法降低token，
所以用三种token组合表达一个item，每一个token codebook是1024，第一个是item类型 第二个是item用途...）

然后先用ar只看text和prompt c1，一直生成c1直到结束（这时候就知道终点了，假如说是75HZ*5s=375token）。再用nar看完整的c1～c8，一次性补全c2～c8.

最后再用decoder把375个c1～c8的8个codebook还原成一段5*24KHZ的音频。

其中codec token就是量化后的结果。

注意，codec和模型不一起训练的（类似于tokenizer），这里可能会有不一致问题。

# Flow Matching

昨天看了好久没看懂，然后摆烂了半天hhhhh

大概就是diffusion的改进版，和vall-E是不同的抽象层级。
```
                  TTS generative model
                         │
             ┌───────────┴────────────┐
             │                        │
       Discrete tokens          Continuous latent
             │                        │
       AR / NAR LM              Diffusion / Flow
             │                        │
          VALL-E              Flow Matching

```
VALL-E主要是负责把语音压缩、量化成token，LLM生成完所有的C1～C8，LLM AR负责补全token，最后flow再把所有的C1～C8离线的embedding全部补成另一种latent embedding，其他组件再把embedding还原成声音。

LLM决定大体上怎么说，flow补充语音细节。

Speech tokenizer 负责把训练语音压缩/量化成离散 speech token；LLM AR 学习 Text → Speech Token；Flow Matching 不是把 token 变成 embedding，而是以 speech-token embedding 为条件，从噪声生成连续的 Mel / acoustic latent；最后 vocoder 把连续声学表示还原成 waveform。

# DITAR
结合LLM AR的长上下文 & Diffusion的局部信息 多模态优势，创造出来。

不用VALL-E量化成8个token，而是直接用VAE把1s24KHZ音频量化成1s->40HZ音频，1HZ64dimension。
这样就能直接做到把1s音频直接做到量化成40个连续的embedding：
```
1s→40×64
```

把4个token合成一个patch，也就是一个patch=100ms，40token->10 * 4* 64(p1)

为了降低计算量，把patch过一遍Aggregation encoder 10 * 4 * 64(p1) -> patch embedding（e1），这样就能降低计算量了。

casual llm: e1 -> e2 -> e3  推理出 h3

下面再由LocDiT(local diffusion transformer):
生成p4也就是[x13,x14,x15,x16]
对于LocDiT而言，patch之间是casual的，patch内部是不mask的。
然后过一遍Aggregation encoder p4 -> e4
直到EOS结束。



