# 15 · Neural Machine Translation by Jointly Learning to Align and Translate（Bahdanau, Cho, Bengio, 2014）

双向 RNN 编码器 + 加性注意力解码器，手写 NumPy。任务：**序列反转**——所以"对齐"有精确真值。

## 论文报告

摘要原文（问题陈述）：

> … we conjecture that the use of a fixed-length vector is a bottleneck in improving the
> performance of this basic encoder–decoder architecture, and propose to extend this by
> allowing a model to automatically (soft-)search for parts of a source sentence that are
> relevant to predicting a target word …

以及关于对齐的那句（本切片把它变成数字）：

> Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model
> agree well with our intuition.

对齐模型（§3.1 的加性打分）：

```
e_ij  = v_a · tanh(W_a s_{i-1} + U_a h_j)
α_ij  = softmax_j(e_ij)
c_i   = Σ_j α_ij h_j
```

论文规模：WMT'14 英法翻译，BLEU 与当时最好的基于短语的系统相当。

## 本仓实测

| 检验 | 实测 |
|---|---|
| 梯度 vs 有限差分（每个参数取前 40 项） | max_abs **5.6e-10** |
| 注意力行和 | 1 − 1.1e-16 |
| 反转任务（长度 5，300 步，3.4 秒） | 精确匹配 **1.00**，对齐命中 **1.00** |
| 只训长度 5 → 测长度 6 | 精确 0.00，对齐 0.89 |
| 混合长度 3–6 训练 → 测长度 7 | 精确 **0.97**，对齐 **1.00** |
| 同上 → 测长度 9 | 精确 0.02，对齐 0.55 |

（对齐命中 = 解码步 t 的注意力峰值落在源位置 L−1−t，即反转的正确对齐。）

## 差异归因

**1. 用 vanilla tanh 循环单元，不是 GRU。** 论文的编码器与解码器都是 GRU。本切片要检验的
是注意力机制，所以把循环单元换成了最简单的形式——但这个替换意味着**不能**把这里的
收敛速度或效果与论文对照。

**2. 没有 BLEU，也没有翻译。** 任务是序列反转，不是英法翻译。好处是对齐有精确真值
（反转的对齐是反对角线），坏处是它比翻译简单得多——"对齐命中 1.00"说明不了论文的
BLEU 结果。

**3. 长度泛化是有限的，而且能看出它学的是什么。** 只训长度 5，长度 6 就归零——模型记住的
是"长度 5 的反转"，不是"反转"这个操作。改成混合长度 3–6 训练后，长度 7 能到 0.97，
但长度 9 崩到 0.02。**注意力图比解码输出退化得慢**（长度 9 时对齐还有 0.55），说明
瓶颈在解码器而不在注意力——但这一条只是观察，没有进一步实验支撑。

**4. "行和 = 1" 是 softmax 的定义，不是学出来的。** 那条测试验证的是实现正确性，
不是论文主张。这一点必须写明，否则读者会把它当成发现。

## 跑

```bash
cd python/15_bahdanau && python3 test_bahdanau.py    # 约 7 秒
```

## 下一步

- `17_annotated_transformer`：把注意力从加性换成点积，接上完整的 Transformer。
