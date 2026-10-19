# 26 · GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism（Huang et al., 2019）

微批流水线：**输出与整批前向逐位相同**，同时峰值激活内存下降。

## 论文报告

摘要原文：

> … we introduce GPipe, a pipeline parallelism library that allows scaling any network that
> can be expressed as a sequence of layers. By pipelining different sub-sequences of layers
> on separate accelerators, GPipe provides the flexibility of scaling a variety of different
> networks to gigantic sizes efficiently.

> Moreover, GPipe utilizes a novel batch-splitting pipelining algorithm, resulting in almost
> linear speedup when a model is partitioned across multiple accelerators.

规模：557M 参数的 AmoebaNet 在 ImageNet-2012 上 top-1 84.4%；6B 参数、128 层
Transformer 在 100+ 语言的语料上训练。

## 本仓实测

4 级流水线（16→32→32→24→8），batch 64，单进程 NumPy 模拟。

| 微批数 M | 峰值激活元素 | 与整批之比 | max\|Δ\|（与整批输出） |
|---|---|---|---|
| 2 | 2,048 | 3.00× | **0.0** |
| 4 | 1,664 | 3.69× | **0.0** |
| 8 | 1,344 | 4.57× | **0.0** |
| 16 | 1,184 | 5.19× | **0.0** |

- 整批前向的峰值是 6,144 个元素。**输出是逐位相同的**（不是 1e-15，是 0.0）。
- 峰值随 M 单调下降，但**收益饱和在流水线深度附近**——16 个微批只给了 5.19×，不是 16×。

11 项检查，0.1 秒。

## 反向传播

之前只验证了前向。`backward_full` / `backward_micro` 补上反向：

| 检验 | 实测 |
|---|---|
| 流水线梯度 vs 有限差分 | max_abs **9.2e-10** |
| 微批梯度 vs 整批梯度（M=2/4/8） | 差 **2.1e-14** |

**拆分批不会改变梯度**，只会改变它如何累加——这是流水线正确性的另一半。

## 重计算

流水线省内存的另一个手段是**不存激活、反向时重放**（checkpointing）。`backward_recompute`
实现了它，并且和存激活的版本给出**完全相同**的梯度（差 0.0）：

| 阶段数 K | 梯度差 | 存储元素（存激活） | 存储元素（重算） | 节省 |
|---|---|---|---|---|
| 4 | 0.0 | 8,192 | 4,096 | 2.0× |
| 8 | 0.0 | 16,384 | 4,096 | 4.0× |
| 16 | 0.0 | 32,768 | 4,096 | **8.0×** |

**重算的存储量与深度无关**（永远是两个张量），所以节省随 K 线性增长。代价是计算：
每层都要从输入重放，复杂度 O(K²)。

## 差异归因

**1. 没有多设备。** 流水线在一个进程里模拟，测的是"同时活着的激活元素数"，不是显存占用。
论文的核心是跨加速器调度，而"almost linear speedup"——**完全没测**。

**2. 重计算补上了。** 不存激活、反向重放，梯度与存激活版本**完全相同**，存储量与深度无关
（K=16 时省 8 倍）。代价是 O(K²) 的重放计算。

**3. 反向传播补上了。** 梯度对有限差分校验到 9.2e-10，微批梯度与整批差 2.1e-14。

**4. 峰值内存用实测，不用解析公式。** "峰值 ≈ 每微批激活 × 流水线级数"这类估计会预测收益随 M
线性增长；实测是饱和的，因为级数只有 4 级。错误估计看起来很有道理，所以这里只报运行得到
的数字。

**5. 规模差得远。** 论文 557M / 6B 参数；这里 4 级、输出 8 维的小网络。

## 跑

```bash
cd python/26_gpipe && python3 test_gpipe.py    # 0.1 秒
```

## 下一步

- `27_scaling_laws`：训练更多参数到底值不值，正好是 GPipe 的动机。
