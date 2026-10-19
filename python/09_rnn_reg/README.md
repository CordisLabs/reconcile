# 09 · Recurrent Neural Network Regularization（Zaremba, Sutskever, Vinyals, 2014）

LSTM + 三种可独立开关的 dropout（输入 / 输出 / **循环连接**），全部手写 NumPy。

## 论文报告

摘要原文：

> Dropout, the most successful technique for regularizing neural networks, does not work
> well with RNNs and LSTMs. In this paper, we show how to correctly apply dropout to
> LSTMs, and show that it substantially reduces overfitting on a variety of tasks.

正文的做法与理由（§2 原文）：

> The main idea is to apply the dropout operator only to the non-recurrent connections.

> Standard dropout perturbs the recurrent connections, which makes it difficult for the
> LSTM to learn to store information for long periods of time. By not using dropout on
> the recurrent connections, the LSTM can benefit from dropout regularization without
> sacrificing its valuable memorization ability.

规模：Penn Tree Bank，medium LSTM 650 单元 / 50% dropout，large 1500 单元 / 65%，
K20 GPU 上半天到一天（Table 1 附近）。

## 本仓实测

语料 6000 字符（Alice，见 `../_shared/data/NOTICE`），hidden 64，4000 步，CPU 约 1 分钟/组。

| 配置 | 最低 val | 结束 val | 曲线（bpc） |
|---|---|---|---|
| 无 dropout | **2.985** | 4.099 | 5.91 → 3.02 → 2.99 → 3.29 → 3.69 → **4.10** |
| 非循环 0.5 | 3.070 | **3.070** | 5.92 → 3.50 → 3.28 → 3.18 → 3.11 → 3.07 |
| 再加循环 0.5 | 3.037 | **3.037** | 5.92 → 3.54 → 3.32 → 3.19 → 3.10 → 3.04 |

结构检验（不训练，毫秒级）：

| 检验 | 实测 |
|---|---|
| 带三种 mask 的解析梯度 vs 有限差分 | max_abs **4.2e-10** |
| mask 丢弃比例 / 缩放 | 0.2998 vs p=0.3；1.428571 = 1/(1−0.3) |
| mask 是按序列固定的（variational） | 形状 (B,dim)，每个时间步用同一个 |
| 循环 mask 清零后，记忆仍在 | 不同首符号仍能传到末步（见下） |

## 训练预算扫描：dropout 的收益随预算增长

之前只报了一个预算（1200 步）。扫三个：

| 步数 | 模型 | 最低 val | 最终 val | 过拟合退化 |
|---|---|---|---|---|
| 1200 | 无 dropout | 3.800 | 4.049 | 0.249 |
| 1200 | dropout 0.5 | 3.871 | 3.871 | **0.000** |
| 2500 | 无 dropout | 3.825 | 5.058 | **1.233** |
| 2500 | dropout 0.5 | 3.850 | 3.872 | 0.023 |
| 5000 | 无 dropout | 3.891 | 7.021 | **3.130** |
| 5000 | dropout 0.5 | 3.852 | 4.061 | 0.209 |

**两个结论**：

1. **论文的主张成立**：没有 dropout 时留出损失在最低点之后持续上升，退化从 0.25 涨到 3.13；
   有 dropout 时曲线基本贴住最低点（退化 ≤ 0.21）。
2. **收益随训练预算增长**：5000 步时两者的退化差 **15 倍**。所以早先"dropout 只买到稳定、
   没买到更高峰值"的说法要补一句——**在这个规模上它的价值体现在长期训练，而不是峰值**。

## 差异归因

**1. 论文的核心主张成立，但价值在长期训练。** 论文说在循环连接上加 dropout 会
损害记忆。实测：非循环 3.070 vs 再加循环 3.037——**加了循环 dropout 反而略好，差异在
种子噪声内**。

原因是结构性的，我用一条不依赖训练的测试固定住了它：**把循环连接清零，记忆仍然能传
下去**。因为 `c_t = f·c_{t-1} + i·g` 这条乘法路径根本不经过 `W_h`，而 `W_h` 只是把
`h_{t-1}` 送进门的输入。所以"在循环连接上做 dropout"伤害的是门的输入，不是细胞状态本身。

这个发现在小规模上尤其明显；在论文的 1500 单元、PTB、几十个 epoch 的设置下，门的输入
被污染可能才是主导效应。**我没有复现那个规模，所以不能断言论文错了**——只能断言在我
这个规模上，它的警告没有表现出来，并且给出了一个可解释的结构性理由。

**2. dropout 买到的是稳定，不是更高的峰值。** 无 dropout 的最低 val 是 **2.985**，比有
dropout 的 3.070 更好——但它在 1600 步后一路涨到 4.099。同预算下的"最终 val"是
**3.07 vs 4.10**。所以正确的说法是"dropout 让曲线停在最低点附近"，而不是"dropout 让
误差更低"。这两句话在论文语境里常被混用，这里分开。

**3. 规模差两个数量级。** 论文 650–1500 单元、PTB（约 100 万词）、GPU 半天到一天；
这里 64 单元、6KB 语料、CPU 一分钟。绝对值不可比，趋势可比。

**4. 判据是我定的。** 论文报 PTB 的 word-level perplexity；这里报自建语料的 bits/char。
两者不能换算，也不该并排比较。

**5. `SYLLABUS.md` 里我原来把「Dropout / Layer Norm / Weight Decay / Early Stopping」
并列为这一篇的内容——这是错的。** 这篇论文只讲 variational dropout；Layer Norm 是
Ba et al. 2016 的另一篇。已改正 SYLLABUS。

## 跑

```bash
cd python/09_rnn_reg && python3 test_rnn_reg.py   # 约 11 秒
python3 report.py                                  # 三组 4000 步，约 3 分钟
```

## 下一步

- 想要复现论文的结论，需要 650+ 单元、PTB 规模与几十个 epoch——这超出当前片段的预算，
  记在这里而不是假装做到了。
- `10_...` 会继续在视觉轨。
