# 02 · A Tutorial Introduction to the Minimum Description Length Principle（Grünwald, 2004）

Bernoulli 源的 **NML（normalized maximum likelihood）后悔值**，全部用 log-sum-exp 算：

```
R(n) = log2 Σ_{k=0}^{n} C(n,k) (k/n)^k ((n-k)/n)^(n-k)
```

## 论文报告

教程的范围（摘要原文）：

> This tutorial provides an overview of and introduction to Rissanen's Minimum Description
> Length (MDL) Principle. The first chapter provides a conceptual, entirely non-technical
> introduction … the second chapter … all the ideas of the first chapter are made
> mathematically precise.

教程里可计算的例子就是这个 NML 码：对每个序列用最大似然，再除以所有序列的似然和，
得到一个合法的概率分布。**教程没有给数字**，所以这一节的"论文报告"是范围说明。

## 本仓实测

| n | R(n)（比特） | R(n) − 0.5·log₂n |
|---|---|---|
| 1 | 1.0000 | +1.0000 |
| 2 | 1.3219 | +0.8219 |
| 10 | 2.2204 | +0.5594 |
| 100 | 3.7236 | +0.4016 |
| 1,000 | 5.3328 | +0.3499 |
| 10,000 | 6.9773 | +0.3334 |
| 100,000 | 8.6330 | **+0.3282** |

四条断言全部通过（0.3 秒）：

| 检验 | 实测 |
|---|---|
| R(n) > 0 | ✅（n=1 时恰为 1 bit：两个序列、各给概率 1）|
| 斜率 = 0.5（n=100…4000 拟合） | 误差 < 0.02 |
| Kraft 和 Σ 2^(−L) = 1（n=12，全枚举） | **1.000000000000000** |
| 偏移收敛到 0.5·log₂(π/2) = 0.3258 | n=1e5 时 **0.3282** |

## 差异归因

**1. 只做了 NML 一个例子。** 教程还讲了 two-part code、prequential MDL、MDL 与 AIC/BIC 的
对比——这些都没做。所以"MDL 是什么"这个问题，本切片只回答了一个侧面。

**2. 教程没有数字可对，所以判据是外部已知结果。** 0.5·log₂(π/2) 这个常数是 Rissanen 对
Bernoulli NML 的已知渐近，不是我从教程里抄的；**我没有核对教程里是否写了它**。这里要说清：
本切片验证的是"实现能复现一个已知渐近常数"，比"跑出个数"强，但不是"复现了教程的结论"。

**3. 实现上踩过一个坑**：`log C(n,k)` 若逐个用乘积算，整体是 O(n²)——n=1e5 时测试要 52 秒。
改成按 k 递推（O(1) 每项）后降到 0.3 秒。这条记在交接文件里。

**4. 与 `01_mdl` 的区别**：01 是 Hinton & van Camp 的权重描述长度，02 是 MDL 理论的
NML 形式。两篇都叫 MDL，但一个是网络正则化，一个是通用编码理论，不能混为一谈。

## 跑

```bash
cd rust && cargo test -p m02_mdl_tutorial && cargo run -p m02_mdl_tutorial --example report_m02_mdl_tutorial
```
