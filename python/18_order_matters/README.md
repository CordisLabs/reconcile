# 18 · Order Matters: Sequence to Sequence for Sets（Vinyals, Bengio, Kudlur, 2015）

两种编码器的对照：集合编码器（逐元素共享映射 + 注意力池化）vs LSTM 编码器（依赖到达顺序）。

## 论文报告

摘要原文（两句，分别对应本切片做了什么、没做什么）：

> In this paper, we first show using various examples that the order in which we organize
> input and/or output data matters significantly when learning an underlying model. We
> then discuss an extension of the seq2seq framework that goes beyond sequences and
> handles input sets in a principled way.

> In addition, we propose a loss which, by searching over possible orders during training,
> deals with the lack of structure of output sets.

论文的任务：语言模型、句法分析、排序数字、估计图模型联合概率。

## 本仓实测

| 检验 | 实测 |
|---|---|
| 梯度 vs 有限差分（两个编码器） | 1.3e-10 / 1.0e-10 |
| **集合编码器**池化上下文的置换差 | **2.8e-17**（机器零） |
| **LSTM 编码器**池化上下文的置换差 | **0.12 – 0.19** |
| 置换输入后解码出的**值序列** | 完全相同（`0.0919 0.2616 0.2985 0.6001 0.7286 0.8142`）|
| 排序 n=6（600 步，约 3 秒） | 集合 **1.000**，LSTM 0.920 |
| 外推 n=8 | 集合 **1.000**，LSTM 0.830 |

结构那条是精确的：集合编码器的池化上下文在 30 次随机置换下最大变化 **2.8e-17**——
它就是不变的。LSTM 编码器同一实验是 0.12–0.19。论文说的 "order matters"，在编码器
这一层就是这两个数字的区别。

## 差异归因

**1. 只做了输入侧，论文的另一半（输出侧的 order loss）没做。** 论文还提出了一个在训练
时搜索输出顺序的损失，用来处理"输出是集合"的情况。本切片只处理了输入侧的不变性，
输出仍按固定顺序（指针解码）。

**2. 任务只有排序。** 论文还有语言模型、句法分析、图模型。

**3. 两个编码器的参数量不同。** 集合编码器没有循环矩阵 `Ue`，所以它更小——"集合编码器
更准"这个结果里，参数量更少这个因素也起作用。只跑了一个随机种子，没有做多种子平均。

**4. 单元是 vanilla tanh，不是 LSTM**（与 15/17/19 一致）。

**5. 池化用的是注意力加权和**，不是论文的 Read-Process-Write 全流程。它抓住了"置换不变"
这个性质，但没有论文完整的编码-处理-解码结构。

## 跑

```bash
cd python/18_order_matters && python3 test_order_matters.py    # 约 7 秒
```

## 下一步

- `20_relational`：关系网络，同样是"对集合的置换不变"的家族。
