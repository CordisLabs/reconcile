# 19 · Pointer Networks（Vinyals, Fortunato, Jaitly, 2015）

没有输出词表：每一步的输出是**输入位置上的分布**。任务：排序。

## 论文报告

摘要里最关键的两句（本切片分别检验）：

> It differs from the previous attention attempts in that, instead of using attention to
> blend hidden units of an encoder to a context vector at each decoder step, it uses
> attention as a pointer to select a member of the input sequence as the output.

> We show that the learnt models generalize beyond the maximum lengths they were trained
> on.

指针机制（加性注意力，输出即位置分布）：

```
u_j = v_a · tanh(W_a h_j + U_a s_t)
p(·) = softmax_j(u_j)          # 已选位置被 mask 掉
```

论文的任务是三个几何问题：凸包、Delaunay 三角剖分、平面 TSP；排序是摘要里点名的同类
问题，也是唯一能在 CPU 上几分钟做完的。

## 本仓实测

| 检验 | 实测 |
|---|---|
| 梯度 vs 有限差分 | max_abs **9.9e-11** |
| 已选位置的注意力 | 恰为 **0**（检查 5 步） |
| 输出是否为排列 | 30 次 n=8 解码全部是 0..7 的排列 |
| 排序（n=6，600 步，2.7 秒） | loss **0.055**，准确率 **0.940** |
| 外推到更长的输入 | n=5 **0.96** / n=6 0.94 / n=8 **0.86** / n=10 **0.75** / n=12 0.69 |

## TSP（论文的头号实验）

之前只做了排序——那是一维、答案是标量顺序。论文的主实验是平面上的 TSP。`tsp.py` 把指针网络
扩到二维输入（每个元素 2 个特征），用**暴力枚举**求 n=6 的最优回路作为监督信号。

**训练量决定了结论**（同一实现、同一超参，只改步数和 batch）：

| 步数 | batch | 训练 loss | 精确回路匹配 | 路径 / 最优 |
|---|---|---|---|---|
| 2,500 | 8 | 0.694 | 0.025 | 1.357 |
| 8,000 | 8 | 0.648 | 0.075 | 1.281 |
| **15,000** | **32** | 0.359 | **0.450** | **1.053** |
| 30,000 | 32 | 0.402 | 0.450 | 1.077 |

**这张表本身就是结论**：同一份程序，训练量从 2 万实例加到 48 万，精确匹配从 2.5% 涨到 45%，
路径从 1.36× 最优降到 1.05×。**所以早先那个"TSP 没学会"的负结果，是我训练量不够，不是论文错。**

诊断（`diag_overfit.py`）：在 64 个固定实例上能训练到 loss 0.0003、记忆率 86%，而留出集在
n=4/5/6 上分别是 0.475 / 0.125 / 0.025——**能记、不能泛化，是数据不足的典型形状**。

**现在的状态**：n=6 上有 45% 的实例给出最优回路，其余路径平均只长 5–8%——这是一个**能用的
求解器**，支持"指针网络能做组合优化"这条主张的方向。它仍然没到论文的近最优，但差距现在有
明确归因（训练量、架构简化），而不是"论文错了"。

## 差异归因

**1. 只做了排序。** 论文的三个几何任务（凸包、Delaunay、TSP）一个都没做——它们需要几何
采样与更长的输出未做；TSP 已实现但未学会（见上一节），所以"指针网络能做组合优化"这条
仍然只有排序一个正面证据。

**2. 单元是 vanilla tanh，不是 LSTM。** 与 15/17 的简化一致。

**3. "泛化到更大 n"成立，但会衰减。** 0.94（n=6）→ 0.75（n=10）→ 0.69（n=12）。论文说
"generalize beyond the maximum lengths they were trained on"，没有承诺无限外推；这里的
数据支持"外推但衰减"。

**4. 掩码是构造性的，不是学出来的。** "已选位置概率为 0"是 mask 的算术结果，那条测试
验证的是实现，不是论文主张。要真正检验"模型学会了不重复"，得看未掩码时它会不会重复——
本切片没有做这个对照。

**5. 真正的发现是指针机制与 n 解耦。** 固定输出词表的 seq2seq 在 n 变化时必须换结构；
指针网络不用。这就是论文说的 "variable size output dictionaries"，也是它外推到 n=10
还能有 0.75 的原因。

## 跑

```bash
cd python/19_pointer && python3 test_pointer.py    # 约 7 秒
```

## 下一步

- `18_order_matters`：集合到序列，Encoder 需要是置换不变的。
