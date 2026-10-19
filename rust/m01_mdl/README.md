# 01 · Keeping Neural Networks Simple by Minimizing the Description Length of the Weights（Hinton & van Camp, 1993）

把「权重的描述长度」当成一笔真实的账来记：二部码（数据项 + 权重项），只扫假定噪声
水平 σ，先验尺度 σ_w 提前固定。零依赖。

## 论文报告

摘要原文：

> Supervised neural networks generalize well if there is much less information in the
> weights than there is in the output vectors of the training cases. So during learning,
> it is important to keep the weights simple by penalizing the amount of information they
> contain. The amount of information in a weight can be controlled by adding Gaussian
> noise and the noise level can be adapted during learning to optimize the trade-off
> between the expected squared error of the network and the amount of information in the
> weights.

§2 的记账方式：**σ_w 提前固定**时，权重的描述长度正比于 Σw²——这就是 L2 惩罚的理论出处。

论文真正的技术贡献是：当输出单元为线性时，**期望平方误差与噪声权重信息量的精确导数
可以高效算出，不需要蒙特卡洛**（摘要最后两句）。

## 本仓实测

数据：真值是三次曲线 `1.5x − 0.8x³`，噪声 σ=0.1，20 个训练点、200 个测试点，
xorshift 固定种子。σ_w = 1（权重 O(1)，这是**事先**定下的，不是搜出来的）。

| 检验 | 判据 | 实测 |
|---|---|---|
| 总码长在 σ 网格上有内点极小 | 不在端点 | ✅ 内点 |
| degree=12：MDL 选的 σ vs 测试最优 σ | 代价 < 1.5× | **1.320×**（MDL 选 0.0866，最优 0.3652）|
| 跨 degree 选择（只用训练数据） | 选中小模型 | **选 degree 5**，测试最优 degree 4，代价 **1.116×** |
| 每个额外权重都要付费 | weight bits 随 degree 严格增 | 4.70 → 21.63 bits（degree 3→15）|
| 确定性 | 两次运行逐位相同 | ✅ |

总码长在 σ=0.0866 处取极小，此时 data bits −30.71、weight bits 17.75。

## 差异归因

**1. MDL 系统性欠正则。** 四个先验尺度下，MDL 选的 σ 恒为 0.0866，而测试最优 σ 分别是
0.1155 / 0.1778 / 0.3652 / 0.7499——**总是偏小**，也就是偏向「假定噪声小、少惩罚权重」。
这不是实现错误，是二部码在有限样本上的已知偏向。

**2. 先验尺度决定结论。** 代价随 σ_w 单调上升：

| σ_w | MDL 选的 σ | 测试最优 σ | 代价 |
|---|---|---|---|
| 0.3 | 0.0866 | 0.1155 | **1.024×** |
| 0.5 | 0.0866 | 0.1778 | 1.150× |
| 1.0 | 0.0866 | 0.3652 | 1.320× |
| 2.0 | 0.0866 | 0.7499 | 1.494× |

**「MDL 能选出最优超参」这句话只在先验尺度选对时成立**；先验选错时，它照样给你一个
看起来很合理的数。这个敏感性必须报，不然就是选择性汇报。

第一版实现把 σ_w 和 σ 耦合（σ_w² = σ²/β），先验可以任意尖锐、码长无下界，跨度选择
直接崩掉（选 degree 15，代价 1.58×）。读到原文 §2「σ_w 提前固定」才发现耦合是我加的，
不是论文的做法。修正后跨度选择立刻正常（1.116×）。

**3. 没有复现论文的技术核心。** 论文的关键贡献是线性输出下精确计算噪声的导数、免蒙特
卡洛。本切片用闭式岭回归，**没有权重噪声**，因此没有那套导数。它复现的是 MDL 的记账
方式，不是那个算法。

## 第二部分：噪声权重（论文 §5）

第一部分复现的是 MDL 的记账方式，没有复现论文的技术核心。这一部分补上：权重本身是
高斯分布 `Q = N(μ, s²)`，先验是 `P = N(0, σ_p²)`。

### 论文报告

§5.1 式 (5)，通信一个权重所需的比特数等于 P 到 Q 的非对称散度：

> the number of bits required to communicate the posterior distribution of a weight is
> equal to the asymmetric divergence (the Kullback-Liebler distance) from P to Q
>
> G(P; Q) = ∫ Q(w) log (Q(w) / P(w)) dw

并且：「high-variance weights are cheaper to communicate but they cause extra variance
in the data mis-fits」——噪声越大越省码，但把数据失拟推高，所以总量有内点最优。

### 本仓实测

两层 tanh 隐单元、线性输出、10 个数据点。权重均值 μ 与 log 标准差 ρ 一起训练。

| 检验 | 判据 | 实测 |
|---|---|---|
| 解析梯度 vs 有限差分 | 全部参数 | max 相对误差 **< 1e-6** |
| 精确期望 vs 蒙特卡洛 | 与抽样标准误比较 | 1e6 样本时 \|diff\|/se = **0.85** |
| KL 非负 | 任意参数 | ✅ |
| 噪声水平是学出来的，不是扫的 | 收敛到内点 | 总码长 **193.43 → 4.85** nats |
| 内点最优 | 统一缩放 σ 后代价上升 | k=1 → 4.85；k=0.5 → 6.19；k=2 → 9.33；k=4 → 51.2 |

学到的 σ 范围 **0.1934 .. 1.0000**；总码长拆成权重码 **3.373 nats** + 数据失拟
**1.475 nats**。

一个观察：输出权重最终是 `v = [0.0, -0.783]`——第一个隐单元被压成 0。它的 σ 恰好停在
**1.0000 = σ_p**，因为一个不起作用的单元只剩 KL 项，而 `KL(Q‖P)` 在 `Q = P`（即
`s = σ_p`）取最小。**这是理论的正确行为，不是初始化残留**，也正好是论文「让网络变简单」
那句话的字面实现。

### 差异归因

**1. 用梯形求积，不是 Gauss-Hermite。** 第一版用 GH，在方差 2.5 处 n=48 时 `E[tanh]`
仍有 4e-6 误差——因为 tanh 的极点在 `±iπ/2`，按 `√(2v)` 缩放后离实轴太近，GH 收敛慢。
这 4e-6 足以让梯度检查来回摆动 1% 量级。改成对高斯权重做梯形求积（被积函数解析且以
`e^{-z²/2}` 衰减，误差按 `e^{-c/h}` 下降），256 个节点到机器精度，顺带也免掉了求根——
**求根漏掉一个节点在权重归一化之后是看不出来的**，这个坑这次差点没发现。

**2. 输出权重不带噪声。** 论文里整层权重都有 Q；本切片只让隐层权重带噪声，输出权重
只付一个高斯先验（相当于 `s = σ_p` 的退化情形）。这是简化，不是复现。

**3. σ_p 仍然是人为定的 1.0。** 第一部分的敏感性结论在这里同样成立：先验尺度决定学到的
噪声水平。

## 跑

```bash
cargo run -p m01_mdl --example report_m01_mdl        # 第一部分：二部码 MDL
cargo run -p m01_mdl --example report_noise  # 第二部分：噪声权重
cargo test  -p m01_mdl
```

## 下一步

- GPipe 式的流水线并行不在本篇范围内；本篇剩下的口子是**完整的噪声层**：输出权重也带 Q。
- 与量化系统的那条线（因子数量 ↔ 过拟合惩罚）可以直接用第一部分的代价函数当惩罚项。
