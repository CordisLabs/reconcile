# 22 · Neural Message Passing for Quantum Chemistry（Gilmer et al., 2017）

消息传递网络：边上的消息 → 节点更新 → 图级读出。

## 论文报告

摘要原文：

> … These models learn a message passing algorithm and aggregation procedure to compute a
> function of their entire input graph. … In this paper, we reformulate existing models into
> a single common framework we call Message Passing Neural Networks (MPNNs) and explore
> additional novel variations within this framework.

> Using MPNNs we demonstrate state of the art results on an important molecular property
> prediction benchmark…

## 本仓实测

| 检验 | 实测 |
|---|---|
| 梯度 vs 有限差分（2 步消息传递） | max_abs **9.4e-11** |
| 图级读出对节点顺序不变（10 次置换） | **5.6e-17** |
| 节点状态随置换等变 | **4.4e-16** |
| 合成图回归（1200 步，1.3 秒） | loss 7.107 → **0.405** |
| 相对"恒预测 0"的基线 | MAE **0.563** vs **1.970**（3.5×） |

不变性与等变性都是**构造性的**：消息聚合用了 `Σ_j adj[i,j]·m(h_i,h_j)` 与 `Σ_i h_i`，
对节点序号的置换天然不变。

## 边特征与消融

之前消息函数只看节点。论文用边特征（QM9 的键类型）。这一节加上边特征，并用一个**答案依赖
边权**的任务做消融：目标是 `Σ_ij adj_ij · e_ij · x_i[0] · x_j[0]`。

| 模型 | 留出集 MAE |
|---|---|
| 无边形特征 | 2.5671 |
| **有边特征** | **1.1710** |

**边特征让误差降到 45%**——它们不是装饰，任务确实需要。注意 400 步时方向是反的（1.94 vs
1.76），1500 步才显出收益：消融也要训够。

## 差异归因

**1. 没有 QM9，也没有任何分子数据。** 论文的核心结果是在 QM9 上预测 13 种分子性质并达到
当时的 SOTA。本切片训练的任务是"图上一个特征之和"——它可精确计算，但不涉及化学。

**2. 边特征补上了，并做了消融。** 无边形特征 MAE 2.567，有边 1.171——边权确实被用上了。

**3. 没有做论文的消融。** 论文比较了多种消息/更新/读出函数的变体，选择了最好的；这里只有
一种（ReLU + sum 聚合 + 求和读出）。

**4. 置换不变是构造性的，不是学出来的**——和 18、20 一样，验证的是实现，不是"模型理解了图"。

## 跑

```bash
cd python/22_mpnn && python3 test_mpnn.py    # 1.3 秒
```
