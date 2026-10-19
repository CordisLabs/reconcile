# 08 · Understanding LSTM Networks（Olah, 2015）

LSTM 单元与 BPTT，手写 NumPy；同一个 copy 任务上同时训练一个 vanilla RNN 作对照。

> 注：这一篇是博客文章，不是论文。

## 论文报告

原文两句（关于细胞状态与遗忘门）：

> The key to LSTMs is the cell state, the horizontal line running through the top of the
> diagram. The cell state is kind of like a conveyor belt. It runs straight down the
> entire chain, with only some minor linear interactions. It's very easy for information
> to just flow along it unchanged.

> This decision is made by a sigmoid layer called the "forget gate layer." It looks at
> `h_{t-1}` and `x_t`, and outputs a number between 0 and 1 for each number in the cell
> state `C_{t-1}`. A 1 represents "completely keep this" while a 0 represents "completely
> get rid of this."

**博客没有给任何基准数字**——它是概念示意图，不是可复现实验。所以这一节的"论文报告"
填的是机制描述，指标全部是本仓自己定的。

## 本仓实测

### 传送带：梯度回到第 0 步的范数

这一步不需要训练，也不需要基准。直接测反向传播时 `∂L/∂h₀` 的范数：

| T | LSTM | vanilla RNN | 倍数 |
|---|---|---|---|
| 1 | 3.68e-02 | 9.99e-02 | 0.4× |
| 5 | 6.34e-03 | 2.72e-03 | 2.3× |
| 10 | 8.63e-04 | 7.20e-05 | 12× |
| 20 | 6.80e-05 | 1.65e-07 | **412×** |
| 40 | 1.33e-06 | 1.34e-12 | **≈ 10⁶×** |

衰减率：vanilla 每步约 1.8×，LSTM 每步约 1.29×。到 T=40，vanilla 的梯度已经等于零
（1e-12）——这就是"信息流不过去"的量化版本。

### copy 任务：同预算对照

任务（Graves 的经典 copy task）：读 T 个符号，一个分隔符，然后复述。答案精确已知，
所以准确率是真数字。两个模型同 hidden=32、同 batch=32、同 lr=3e-3、同步数。

| T | 步数 | LSTM 准确率 | vanilla 准确率 | 随机 |
|---|---|---|---|---|
| 5 | 1500 | 0.872 | **1.000** | 0.100 |
| 10 | 4000 | 0.717 | **0.983** | 0.100 |

单批过拟合（32 条固定样例，3000 步）：loss 0.0182，准确率 **1.000**。

## 训练预算扫描：vanilla 的优势是预算效应吗？

之前只报了一个预算（1500 步）。扫一遍：

| T | 步数 | LSTM | vanilla |
|---|---|---|---|
| 5 | 1500 | 0.872 | **1.000** |
| 5 | 8000 | 0.993 | **1.000** |
| 10 | 1500 | 0.513 | **0.860** |
| 10 | 8000 | 0.809 | **0.996** |
| 20 | 1500 | 0.430 | **0.538** |
| 20 | 8000 | 0.470 | **0.876** |
| 50 | 4000 | **0.102** | **0.102** |

**结论分两层**：

1. **T ≤ 20：vanilla 在每个预算下都更好**——这不是预算效应，加步数后差距缩小但方向不变。
2. **T = 50：两个都在随机水平**（0.102 ≈ 1/10）。即使把隐藏层从 32 加到 128，LSTM 也学不会
   T=50 的 copy。

**第二层是"程序不够格"，不是论文的结论**：论文关心的正是长延迟。我的 LSTM 在 T=50 上完全
学不会，说明训练量或实现还差得远，**因此没有检验"LSTM 在长延迟上更好"这条主张**。

梯度保留的测量（T=20 时 412×、T=40 时 ≈10⁶×）是精确的，不受此影响——但**梯度保留不等于
任务表现**，这一点在 T ≤ 20 的数据里看得很清楚。

## 差异归因

**1. 机制成立，基准优势没复现——这是本篇最重要的结论。** T=20 时 LSTM 的梯度回收是
vanilla 的 412 倍，但同一预算下 vanilla 复制得更好（T=5：1.000 vs 0.872；T=10：
0.983 vs 0.717）。

原因不神秘：**梯度保留是长延迟的必要条件，不是短延迟的充分条件**。T=5/10 时 vanilla
的梯度还有 7.2e-5，完全够用；而它的参数量只有 LSTM 的 1/4（1408 vs 5632），在这个
小规模上更好优化。LSTM 的优势区间应该出现在 vanilla 梯度真正消失的地方（T≥40），
而那个规模在 CPU + NumPy 的预算内训不动。

所以本切片验证的是**机制**（传送带存在），不是**众所周知的结论**（LSTM 总比 RNN 好）。
后者在这个规模上不成立，写在这里，而不是悄悄换个任务让它成立。

**2. forget bias 是生死线。** 遗忘门偏置若为 0，`f = sigmoid(0) = 0.5`，细胞状态每步
衰减一半，T=20 后是 0.5²⁰ ≈ 1e-6——**训练开始之前记忆就已经没了**。加上 `bias=1`
（`f ≈ 0.73`）后，T=20 的 copy 准确率从 0.092（= 随机）升到 0.396。这是单元能否学习
长延迟的分界线，而且它是一行初始化，不是架构差别。

**3. 单批能背下来（acc 1.000）不等于学会了任务。** 3000 步把 32 条样例背到完美，
在 256 条新样例上却只有 0.717。这是过拟合，不是能力——两者必须分开。

**4. copy task 不是博客里的东西。** 博客用语言模型举例，没有可跑的任务；copy task 是
Graves 等人的标准探针。用它是因为答案可精确判定，不是因为博客用了它。

## 跑

```bash
cd python/08_lstm && python3 test_lstm.py    # 约 7 秒
python3 report.py                             # 梯度表 + 对照表，约 34 秒
```

## 下一步

- 训练轨的 LSTM 版本要真正体现优势，需要 T≥40 与更大预算；这超出当前片段的范围，
  记在 `SYLLABUS.md` 第 08 条的备注里。
- 第 09 条（RNN 正则化）会在这个单元上继续加 dropout 与 layer norm。
