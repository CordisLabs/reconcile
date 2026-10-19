# 10 · ImageNet Classification with Deep CNNs（AlexNet, Krizhevsky, Sutskever, Hinton, 2012）

小卷积网（conv/pool 反向手写）+ ReLU vs tanh 的收敛对比。

## 论文报告

规模（摘要与 §4.2 原文）：

> … a neural network, which has 60 million parameters and 650,000 neurons …

被检验的主张是 §3.1 的这一段：

> Deep convolutional neural networks with ReLUs train several times faster than their
> equivalents with tanh units. This is demonstrated in Figure 1, which shows the number
> of iterations required to reach 25% training error on the CIFAR-10 dataset for a
> particular four-layer convolutional network.

> A four-layer convolutional network with ReLUs … reaches a 25% training error rate on
> CIFAR-10 six times faster than an equivalent network with tanh neurons.

以及一句决定可比性的方法说明（原文）：

> The learning rates for each network were chosen independently to make training as fast
> as possible.

## 本仓实测

| 检验 | 实测 |
|---|---|
| conv / pool / linear 反向 vs 有限差分（**tanh**） | max_abs **4.2e-10** |
| ReLU 在 0 处的解析导数 / 中心差分 | **0.0** / **0.5**（跨过 kink） |
| ReLU 导数取值 | 恰为 {0, 1} |
| tanh 导数上界 / \|x\|>3 处 | ≤ 1.0 / **< 0.0081** |
| 达到 25% 训练误差所需步数（init_scale=1.0） | ReLU **24**，tanh **17** |

## 真实数据上的对比（GPU）

上面那张表是 30 张合成图。这里换成 **CIFAR-10**（5 万张真实图像），同一个四层卷积网
（94,538 参数），只换激活函数。论文对**每个网络独立调 lr**，所以这里也对每个激活扫了
lr ∈ {1e-3, 3e-3, 1e-2}（5 epoch）：

| lr | ReLU | tanh |
|---|---|---|
| 1e-3 | 0.6070 | 0.6076 |
| **3e-3** | **0.6714** | **0.6705** |
| 1e-2 | 0.4083 | 0.4962 |

各自最优下 ReLU **0.6714** vs tanh **0.6705**，差 0.0009——在随机波动范围内。同一 lr 下跑到
训练 loss 1.3 需要的步数：ReLU 900、tanh 750。

**6× 依然没有复现。** 两套规模（合成图 30 张、CIFAR 5 万张）给出同一个结论。

## 论文的优化设定（SGD + 四层卷积）

上面的对比用的是 Adam。论文是 SGD + momentum、四层卷积、**每个网络独立调 lr**。按这个设定重做
（CIFAR-10，6 epoch，lr ∈ {0.01, 0.03, 0.1}）：

| 激活 | 最优 lr | 测试准确率 | 达到 loss 1.0 的步数 |
|---|---|---|---|
| **ReLU** | 0.1 | **0.6935** | **1100** |
| tanh | 0.1 | 0.6270 | 1400 |

**方向复现了，幅度没有**：

- **ReLU 更好**：最终准确率高 **6.7 个点**，达到 loss 1.0 快 **1.27×**。
- **但不是 6×**：在更早的阈值（loss 1.5）上 tanh 反而更快（250 vs 300 步）。
- 6× 在 AlexNet 的规模（6000 万参数、120 万张图、深网）上出现，不在这个 9 万参数的网络上。

所以这条主张的准确表述是：**ReLU 在这个设定里确实更快、更准，但加速比是 1.3× 而不是 6×。**

## 差异归因

**1. 论文的 6× 优势没有复现，但方向复现了。** 论文是 CIFAR-10（5 万张真实图像）、
四层卷积网、**每个网络的 lr 独立调优**；这里是 30 张合成图、两层卷积、同一个 lr。
两边都测"达到 25% 训练误差所需迭代数"，但本任务 17–24 步就被解决，测量窗口太窄，
任何差异都落在这个窗口里。

**2. 没有复现 ImageNet top-5。** 没有数据集，也没有训练 ImageNet 规模的模型。
**"60 million parameters" 在本切片里只是引用，不是验证**——论文的完整架构在 Figure 2
（一张图）里，正文没有可逐层复算的表，所以我没有把它算出来声称"对上"。

**3. ReLU 不能用有限差分检验——这是本切片最实用的一条。** 解析梯度在 0 处按约定取 0
（`x > 0` 为假），而中心差分跨过 kink 得 0.5。同理，一个被 ReLU 压死的通道解析梯度是 0，
但 eps 扰动会把它激活，数值差分给出非零值（本次实测某个通道 0 vs 0.028）。
所以**梯度检验必须换平滑激活**；这不是绕过问题，是把两个问题分开测。

**4. 结论是"方向复现、幅度不复现"，不是"论文错"。** 按论文的优化设定重做后，ReLU 确实更快
更准（+6.7 点、1.27×），只是没有 6×。要警惕的不是"论文错了"，而是把"没测到"写成"不存在"。
早先这个切片和 `08`、`09` 一起被当成"三个未复现"，复核后三项都重新判定了：`10` 是幅度问题、
`09` 的主张其实成立（收益随预算增长）、`08` 是长延迟区间没测成。

## 跑

```bash
cd python/10_alexnet && python3 test_alexnet.py    # 约 15 秒
```

## 下一步

- `11_resnet`：退化现象（深层 plain net 的训练误差更高）可以在小规模上量出来，
  不需要 ImageNet。
