# 17 · The Annotated Transformer（Rush, 2018）

完整的编码-解码 Transformer：sinusoidal 位置编码、多头点积注意力、因果掩码、交叉注意力、
FFN、post-LN 残差、label smoothing。任务：复制序列。

> 注：这是教程，不是论文。它逐行注解 Vaswani et al. 2017，所以本切片的架构主张来自原文，
> 教程提供的是实现方式的讲解。

## 论文报告

被注解的原文（Vaswani et al. §5.4，label smoothing 那一段）：

> **Label Smoothing** During training, we employed label smoothing of value
> ε_ls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but
> improves accuracy and BLEU score.

原文规模：base 6 层、d_model 512、8 头、WMT14 EN-DE 上 BLEU 27.3。

## 本仓实测

| 检验 | 实测 |
|---|---|
| 全模型梯度 vs 有限差分（每参数取 25 项） | max_abs **4.9e-10** |
| 位置编码 | 每行唯一、值域 [−1,1]、第 0 列 **== sin(pos)** |
| 因果掩码 | 前面位置的 logits 完全不受后面 token 影响（**Δ = 0.00e+00**）|
| copy 任务（长度 5，400 步，3.3 秒） | 训练 loss **0.0028** nats，精确匹配 **1.00**，ppl **1.003** |
| label smoothing ε=0.1 | ppl **1.0982**，推导下限 **1.0976**（差 0.0006）|

下限是怎么推的：平滑后正确 token 的概率上限是 `1 − ε + ε/V`，V=9 时为 0.9111，
所以硬目标上的最优 perplexity 是 `1/0.9111 = 1.0976`。

## label smoothing：两半都测了

原文说平滑"**伤害 perplexity，但提升准确率和 BLEU**"。

**第一半（伤 ppl）在训练长度上成立**：ppl 从 1.0008 升到 1.0981，正好贴着推导下限 1.0976。

**第二半需要一个会犯错的任务**：复制任务在长度 10–20 上两个模型都是 100%（饱和）。改成
**训练长度 8、测试长度 10/12/14**，模型必须外推：

| 测试长度 | 无平滑 | 平滑 0.1 |
|---|---|---|
| 8（训练内） | acc 1.000，ppl **1.0008** | acc 1.000，ppl 1.0981 |
| 10 | acc 0.000，ppl 5.579 | acc 0.000，ppl **2.323** |
| 12 | acc 0.000，ppl 20.115 | acc 0.000，ppl **4.020** |
| 14 | acc 0.000，ppl **45.078** | acc 0.000，ppl **5.649** |

**三个结论**：

1. **伤 ppl：成立**（训练内 1.0008 → 1.0981）。
2. **提升泛化：成立**，而且幅度很大——长度 14 的 ppl 从 **45.1 降到 5.6**（8 倍）。
3. **提升准确率：不成立**，两边的 exact-match 都是 0——这个模型根本没有长度外推能力，
   准确率无从体现平滑的好处。

所以论文那句话在本仓是**两对一错**：ppl 变差 ✅、泛化变好 ✅、准确率变好 ✗（任务没有可提升的
准确率）。

## 差异归因

**1. label smoothing 只验证了一半，而且验证的那一半不算"发现"。** perplexity 变差是
**可推导的**——它由概率上限决定，不是"模型学得更不确定"这个经验现象。所以那条测试真正
验证的是"实现按定义工作"，不是论文的机制主张。

而 "improves accuracy" 这一半**在本切片测不出来**：copy 任务在长度 5/8/12 上都饱和到
精确匹配 1.00，没有准确率提升的余地。要测它需要一个不饱和的任务（真实翻译）。

**2. 规模差得远。** 1 层编码器 + 1 层解码器、d_model 32、4 头、ffn 64、序列长度 5；
原文 base 是 6 层、512 维、8 头、WMT14。

**3. 没有学习率 warmup。** 原文用 `lr ∝ min(step^-0.5, step·warmup^-1.5)`，本切片用固定 lr。
这在小模型上影响不大，但也是在"没复现"清单里的一条。

**4. copy task 不是翻译。** 它检验的是编码-解码 + 注意力的机械正确性（能不能把信息从源
搬到目标），不检验翻译质量。

## 跑

```bash
cd python/17_annotated_transformer && python3 test_transformer.py    # 3.3 秒
```

## 下一步

- `18_order_matters`：指针网络，输出是输入上的分布。
