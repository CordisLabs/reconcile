# 07 · The Unreasonable Effectiveness of Recurrent Neural Networks（Karpathy, 2015）

字符级语言模型：vanilla RNN，前向与 BPTT 都是手写的 NumPy，无框架。

> 注：这一篇是博客文章，不是论文。这也是原始清单里唯一的"非论文"——一并实现，
> 因为它给了 RNN 最干净的教学式定义。

## 论文报告

博客里教程代码的更新式（原文代码）：

```python
self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
y = np.dot(self.W_hy, self.h)
```

训练：mini-batch SGD + RMSProp 或 Adam；测试时"采样一个字符、喂回去、再采样"。

文中展示效果的 5 个实验的规模：
- Paul Graham：约 1MB 语料，2 层 LSTM 512 hidden（≈3.5M 参数），dropout 0.5，
  batch 100，BPTT 100 字符，Titan Z 上一个 batch 约 0.46 s
- Shakespeare：4.4MB，3 层 LSTM 512 hidden，训练几小时
- 另有 Wikipedia 96MB、LaTeX 16MB、Linux 源码、8000 个婴儿名

**两个必须说清的事实**：
1. **原文没有给出 bits/char 这个指标**。所以本节"论文报告"填的是做法与规模，不是数字。
2. **文中所有展示效果用的都是 LSTM**（原文：*"From here on I will use the terms
   RNN/LSTM interchangeably but all experiments in this post use an LSTM"*）。教程里
   那个 vanilla RNN 只是讲解用的。

## 本仓实测

语料：Alice's Adventures in Wonderland 前 20,000 字符（公有领域，Project Gutenberg
#11，见 `../_shared/data/NOTICE`），67 个唯一字符，前 90% 训练 / 后 10% 验证。

| 检验 | 判据 | 实测 |
|---|---|---|
| BPTT 解析梯度 vs 有限差分 | 全部参数 | max_abs **2.6e-10**，max_rel **5.9e-7** |
| 周期 3 复读任务（能学会吗） | loss < 0.05 nats | **0.0065 nats**，贪心解码完全复现 `bcabcabca` |
| 留出 bits/char vs order-0 | 低 1 bit 以上 | **2.952** vs **4.495**（低 1.54 bit）|
| 留出 bits/char vs 均匀分布 | 低于 log₂67 | **2.952** vs **6.066** |

容量消融（各 2000 步，CPU）：

| hidden | 训练时间 | 训练 bits/char | 留出 bits/char |
|---|---|---|---|
| 16 | 6.8s | 2.890 | 3.117 |
| 32 | 8.8s | 2.570 | 3.000 |
| 64 | 9.2s | 2.134 | **2.952** |

采样（hidden=64，temperature 0.7，用正文前 24 字符起头）：

```
it ary she Rabbit mice she, “and she wnine her fant of child be notring,
that’ve the rind to souppy: fur think I I shat inlo gried on it over near!”
 *     *      *      *      *      *      *      *      *      *      *
```

能看出人名、引号、说话格式、甚至原文里那行 `* * * * *` 分节符。

## n-gram KL 曲线

清单里的第二条指标，之前没做：每训练 150 步采样 4000 字符，比较采样文本与语料的 n-gram
分布（`ngram_kl.py`，1.7 秒）。

| 步数 | KL(1) | KL(2) | KL(3) |
|---|---|---|---|
| 150 | 0.0524 | 1.4991 | 8.1289 |
| 300 | 0.0763 | 1.1923 | 6.3597 |
| 450 | 0.0714 | 1.0586 | 5.6995 |
| 600 | 0.0997 | 1.0309 | **5.2585** |

**unigram 一开始就对齐（0.05），高阶统计量缓慢逼近**：trigram 从 8.13 降到 5.26。
这正是字符级 RNN 该有的学习顺序——先学字母频率，再学局部组合。

## 差异归因

**1. 模型类别不同，这是最重要的一条。** 我们实现的是 vanilla RNN，博客展示效果用的是
LSTM，而博客自己说 LSTM *"works slightly better in practice"*。所以这条采样**不能**拿去
跟博客里的莎士比亚样本比质量——那是在比两个不同的模型类。

**2. 规模差三个数量级。** 博客 4.4MB 语料 / 3 层 512 hidden / GPU 上几小时；这里 20KB /
1 层 64 hidden / CPU 上 9 秒。想复现博客的质量，这两项都得先涨上去。

**3. 留出误差有明显过拟合缺口。** 训练 2.13 vs 留出 2.95 bits/char。20KB 语料对 64 个
hidden 来说太小了；博客里那种 1MB 语料 + dropout 才是它的正确用法。

**4. 判据是我自己定的，不是原文的。** 原文没有 bits/char 指标；order-0 熵与 log₂67
是**我为这个仓库加的**判据，用来回答"模型到底有没有用上下文"。这个区别必须写明，
否则读者会以为是原文的结论。

**5. 语料没有用 tiny-shakespeare。** `SYLLABUS.md` 原本写的是 tiny-shakespeare，
但那份语料不进仓库（来源与体积都不合适）。改成 20KB 的 Alice 片段：公有领域、
体积可控、离线可复现。它比 tiny-shakespeare 小 50 倍，所以上面的绝对值**不能**直接
跟任何 tiny-shakespeare 的公开结果对照。

## 跑

```bash
cd python/07_char_rnn && python3 test_rnn.py     # 约 9 秒
python3 report.py                                 # 消融表与采样，约 25 秒
```

## 下一步

- ~~采样文本的 n-gram KL 曲线~~ —— 已补（见上一节），trigram 8.13 → 5.26。
- LSTM 版本见 `08_lstm`。
