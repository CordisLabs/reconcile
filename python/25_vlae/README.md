# 25 · Variational Lossy Autoencoder（Chen et al., 2016）

标准 VAE 的**后验坍缩**：KL 权重越大，隐变量承载的信息越少，活跃维度越少。

> 注：只实现了 VAE 与坍缩测量，**论文的核心改进（自回归先验 / 有损编码 / IAF）没有实现**。

## 论文报告

摘要原文：

> Our proposed VAE model allows us to have control over what the global latent code can
> learn and, by designing the architecture accordingly, we can force the global latent code
> to discard irrelevant information such as texture in 2D images, and hence the VAE only
> "autoencodes" data in a lossy fashion.

> … by leveraging autoregressive models as both prior distribution p(z) and decoding
> distribution p(x|z), we can greatly improve generative modeling performance of VAEs,
> achieving new state-of-the-art results on MNIST, OMNIGLOT and Caltech-101 Silhouettes …

**注意**：抽象里**没有**"后验坍缩"这个词——那是正文的问题动机；摘要强调的是"可控的有损
编码 + 自回归"。

## 本仓实测

6 维两簇高斯数据，隐变量 8 维，300 步，每个 β 一次训练。

| β | KL | 活跃维度（KL > 0.01） | 重建误差 |
|---|---|---|---|
| 0.01 | **10.242** | 8/8 | 0.190 |
| 0.10 | 3.750 | 8/8 | 0.415 |
| 1.00 | 1.181 | 6/8 | 0.851 |
| 10.0 | **0.102** | **4/8** | 5.838 |

- KL 随 β **单调下降**（10.242 → 0.102）。
- 活跃维度 8 → 4：一半的隐变量**完全不再编码信息**。
- 重建误差从 0.190 涨到 5.838：坍缩的代价就是重建变差。

5 项检查，0.3 秒。

## 自回归流先验（GPU）

上面测的是**动机**（后验坍缩）。这一节补上论文的**解法**：把对角高斯先验换成一个自回归流
（MADE + 仿射变换），编码器、解码器、隐变量维度、优化器和数据完全相同，只有先验不同。

MNIST，隐变量 16 维，Bernoulli 解码器，3 个 epoch：

| 先验 | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| 对角高斯 | 0.2474 | 0.2350 | **0.2290** |
| **自回归流** | 0.1438 | 0.1150 | **0.1098** |

**流先验把 bits/dim 减半（0.229 → 0.110）。** 变化量的雅可比符号单独有测试（1 维仿射流的
闭式密度），因为符号写反会让 bits/dim 变成负数——第一版确实是这样。

**注意**（流先验那一节）：绝对值和论文不可比（论文是 64 维隐变量、训练到收敛、报的是 bits/dim 的完整实现；
这里是 16 维、3 个 epoch、单样本 ELBO）。可比的是**同一协议下两种先验的差距**。

## 论文的两个部件：自回归解码器 + 自回归先验

VLAE 的核心是"解码器和先验都自回归"，这样隐变量只需携带全局结构。两个部件分别和一起测
（MNIST，隐变量 16 维，2 个 epoch，其余完全相同）：

| 配置 | bits/dim |
|---|---|
| 独立 Bernoulli 解码器 + 高斯先验 | 0.2350 |
| 自回归解码器 + 高斯先验 | 0.1719 |
| **自回归解码器 + 流先验（VLAE）** | **0.0945** |

**两个部件各自有效，叠起来更好：0.2350 → 0.0945，降了 60%。**

自回归解码器是**逐行**的（第 t 行依赖 z 和第 t−1 行），不是 PixelCNN 的逐像素——这是一个
廉价的代理，但它把"解码器建模局部结构"这件事测出来了。

## 差异归因

**1. 核心改进两个都做了：自回归流先验 + 自回归解码器。** 摘要的两条——可控的有损编码、自回归先验/解码器（以及
正文的 IAF）——本切片完全没有。它测量的是**动机**，不是解法。

**2. 用 β 扫出坍缩，而不是论文的机制。** 论文不是用 β 控制坍缩的，它靠架构设计；这里用 β
是因为它能在几秒内把现象暴露出来。所以这条曲线说明的是"KL 权重与信息量的权衡"，不是
"VLAE 解决了坍缩"。

**3. 数据是合成的**（两簇高斯），不是 MNIST / OMNIGLOT / Caltech-101。

**4. 指标是 MSE，不是 bits/dim。** 论文报的是密度估计的 bits/dim，两者不能换算。

**5. 没有测"活跃维度与下游任务的关系"**，而这正是论文"全局表示"主张的落点。

## 跑

```bash
cd python/25_vlae && python3 test_vlae.py    # 0.3 秒
```
