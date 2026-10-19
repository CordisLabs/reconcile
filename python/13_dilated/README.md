# 13 · Multi-Scale Context Aggregation by Dilated Convolutions（Yu & Koltun, 2015）

单通道 2D 卷积 + 膨胀（把卷积核"撑开"），论文里的感受野数字全部用一次前向量出来。

## 论文报告

摘要原文：

> The module is based on dilated convolutions, which support exponential expansion of
> the receptive field without loss of resolution or coverage.

Figure 1 与感受野公式（原文）：

> (c) F3 is produced from F2 by a 4-dilated convolution; each element in F3 has a
> receptive field of 15×15. The number of parameters associated with each layer is
> identical. The receptive field grows exponentially while the number of parameters grows
> linearly.

> … the size of the receptive field of each element in `F_{i+1}` is
> `(2^{i+2} − 1) × (2^{i+2} − 1)`.

§3 的 context module 规格（原文）：

> The basic context module has 7 layers that apply 3×3 convolutions with different
> dilation factors. The dilations are 1, 1, 2, 4, 8, 16, and 1.

> Since the receptive field of the context network is 67×67, we pad the input feature
> maps by a buffer of width 33.

还有一条实用的警告（原文）：

> Our initial attempts to train the context module failed to yield an improvement in
> prediction accuracy. … we found that random initialization schemes were not effective
> for the context module.

## 本仓实测

不需要数据集、不需要训练，全是算术。16 项检查，0.3 秒。

| 检验 | 实测 |
|---|---|
| 膨胀卷积 ≡ 零插入核的普通卷积（d=1,2,4,8） | max\|Δ\| = **0.0** |
| 堆叠 [1] / [1,2] / [1,2,4] 的感受野 | 3×3 / 7×7 / **15×15**（= Figure 1） |
| 7 层 `1,1,2,4,8,16,1` 的感受野 | **67×67**（= 论文，故 padding 33） |
| 指数公式 n=1..4 | 3 / 7 / 15 / **31** = 2^(n+1)−1 |
| 覆盖无洞 | 15×15 内 **225/225** 个位置非零 |
| 参数量 | 7 层 3×3 = **63**，等效单个 67×67 = **4489** |
| 分辨率 | 输出形状 = 输入形状 |

## 差异归因

**1. 只验证架构算术，没有验证分割效果。** 论文在 PASCAL VOC 2012 上报了 front-end
69.8% mIoU（val）/ 71.3%（test），以及插入 context module 后的提升。本切片没有数据集、
没有训练、没有 IoU——**这些数字一个都没复现**。它验证的是"这个模块的感受野确实是 67×67"，
而不是"这个模块让分割更准"。

**2. 省略了 ReLU 与 1×1 投影。** 论文的每层是 `3×3×C 卷积 → ReLU`，最后还有 1×1 投影。
它们不改变感受野大小，但改变"感受野里装的是什么"。省掉它们不影响本切片的结论，但必须
写明，否则读者会以为这就是完整模块。

**3. 没有实现论文的 identity 初始化**（式 4：`k_b(t,a) = 1[t=0]1[a=b]`）。论文明确说
随机初始化训不动 context module——这是全文最实用的警告，但本切片没有训练，所以既没复现
也没验证它。

**4. 这是少数能完全对上的论文数字。** 67×67 与 padding 33 是自洽的
（(67−1)/2 = 33），且我用一次前向量出来而不是抄公式。同类"论文感"最强的数字，往往就是
这种纯算术的。

## 跑

```bash
cd python/13_dilated && python3 test_dilated.py    # 0.3 秒
```

## 下一步

- `12_cs231n` 的 im2col 卷积可以用同样的精确等价来验证（im2col 卷积 == 朴素循环卷积）。
