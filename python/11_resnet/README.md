# 11 · Deep Residual Learning for Image Recognition（He, Zhang, Ren, Sun, 2015）

plain 与残差 MLP 的对照，检验论文的第一个结论：**退化现象**。

## 论文报告

摘要第一句：

> Deeper neural networks are more difficult to train.

退化现象（引言原文）：

> When deeper networks are able to start converging, a degradation problem has been
> exposed: with the network depth increasing, accuracy gets saturated (which might be
> unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused
> by overfitting, and adding more layers to a suitably deep model leads to higher
> training error …

论文总结的两条（引言原文）：

> 1) Our extremely deep residual nets are easy to optimize, but the counterpart "plain"
> nets (that simply stack layers) exhibit higher training error when the depth increases

其理由（§3.2 原文）：

> … if the added layers can be constructed as identity mappings, a deeper model should
> have training error no greater than its shallower counterpart.

论文规模：ImageNet 上 152 层残差网，**3.57% top-5 错误率**，ILSVRC 2015 第一；
CIFAR-10 上另做了 100 层与 1000 层的分析。

## 本仓实测

10 类合成高斯簇，150 个样本，width 32，400 步 Adam，两个种子一致。

| blocks（层数） | plain 训练准确率 | residual 训练准确率 |
|---|---|---|
| 4（8 层） | 1.000 | 1.000 |
| 8（16 层） | 1.000 | 1.000 |
| 16（32 层） | 0.860 | 1.000 |
| **24（48 层）** | **0.100**（= 随机） | **0.993** |
| 32（64 层） | 0.127 | 0.987 |

梯度校验：plain **2.1e-10**，residual **2.5e-10**（对有限差分）。

24 block 的 plain 网络不是"差一点"，是**完全没有训起来**——训练准确率等于 10 类随机。
这正是论文说的"degradation is not caused by overfitting"：训练误差本身在上升。

## 真实数据上的深度扫描（GPU）

上面是合成数据上的 24 块 MLP。这里换成 **CIFAR-10**，plain 与 residual 同样宽度（32 通道），
6 epoch，只改深度：

| 深度 | plain | residual |
|---|---|---|
| 10 层（5 块） | 0.6587 | 0.7003 |
| 20 层（10 块） | 0.6636 | 0.7359 |
| 40 层（20 块） | **0.5599** | **0.7364** |

plain 从 20 层加到 40 层，准确率**掉了 10 个百分点**（0.6636 → 0.5599），而 residual 单调
不降（0.7003 → 0.7359 → 0.7364）。**退化现象在真实数据上复现了**，这也是 He et al. 的原始
观察：加层不是"学得更好或不变"，而是**更差**。

## 差异归因

**1. 用 MLP，不是卷积网。** 论文是 ImageNet 上的 ResNet-152。退化现象是**深度**现象，
不依赖卷积；但本切片没有复现 ImageNet 的任何数字（152 层、3.57%）。

**2. 没有 BatchNorm。** 论文的 plain 网络带 BN。本切片的退化更剧烈（24 block 直接停在
随机水平），但**不能据此说"没有 BN 时退化更严重"**——那是另一个设定，一个数据点撑不起
这个结论。要说清楚：我只是没有实现 BN。

**3. 判据是训练误差，不是测试误差。** 这正是论文关心的量（"higher training error"）。
测试集在这里没有意义：我们要看的是"能不能拟合"。

**4. 这次是复现，与前三个切片相反。** 08（LSTM 优势）、09（dropout 防过拟合）、
10（ReLU 6× 加速）都没复现，11 复现了。区别不在运气，在**主张的类型**：

- **结构性主张**（深网络难优化、感受野是算术、梯度衰减速率）在小规模上就能验证；
- **规模相关的经验主张**（6 倍加速、dropout 带来的泛化收益）依赖论文的具体规模，
  砍两个数量级后就不再成立。

这条区分值得写进交接文件。

## 跑

```bash
cd python/11_resnet && python3 test_resnet.py    # 2.7 秒
```

## 下一步

- `12_resnet_v2`：pre-activation（BN→ReLU→Conv）对深层梯度的影响，同样可以在 MLP 上量。
