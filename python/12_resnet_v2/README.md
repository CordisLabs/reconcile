# 12 · Identity Mappings in Deep Residual Networks（He, Zhang, Ren, Sun, 2016）

ResNet v1（加和后激活）与 v2（pre-activation，shortcut 保持恒等）的对照，测量**回到输入的梯度**。

## 论文报告

摘要原文（也是本切片检验的那句话）：

> … we analyze the propagation formulations behind the residual building blocks, which
> suggest that the forward and backward signals can be directly propagated from one block
> to any other block, when using identity mappings as the skip connections and
> after-addition activation.

规模（摘要原文）：

> We report improved results using a 1001-layer ResNet on CIFAR-10 (4.62% error) and
> CIFAR-100, and a 200-layer ResNet on ImageNet.

## 本仓实测

两种块的定义：

```
pre  (v2):  h = h + W2·relu(W1·h)          # shortcut 是纯恒等
post (v1):  h = relu(h + W2·relu(W1·h))    # shortcut 上带激活
```

`‖∂L/∂x‖`（回到输入的梯度范数），固定输入、不训练：

| blocks（层数） | pre | post | 比值 |
|---|---|---|---|
| 4（8 层） | 4.13e+00 | 3.42e+00 | 1.2× |
| 8（16 层） | 4.32e+01 | 6.79e+00 | 6.4× |
| 16（32 层） | 4.22e+03 | 3.65e+02 | 11.6× |
| 32（64 层） | 4.65e+07 | 3.18e+04 | **1461×** |
| 64（128 层） | 3.86e+14 | 3.44e+10 | **11214×** |

梯度校验（两个变体都对有限差分）：pre **2.7e-10**，post **2.8e-10**。

## BatchNorm：梯度爆炸的真正原因

上面测到 64 块时 pre-activation 的输入梯度范数是 4.1e14。但那是因为**这个实现没有 BatchNorm**，
不是 pre-activation 本身的性质。这一节把 BN 加上（前向 + 反向，梯度对有限差分校验到 3.3e-10）：

| 变体 | 64 块时 ‖dL/dx‖ | 加 BN 后 |
|---|---|---|
| pre-activation | 3.86e14 | **5.25e01** |
| post-activation | 3.44e10 | **2.60e02** |

**BN 把爆炸消掉了。** 所以之前的"梯度爆掉"是缺 BN 的后果，而不是 pre-activation 的缺陷——
这正是 ResNet v2 论文要澄清的一点：identity mapping 要配上 BN 才成立。

## identity 初始化（式 4）

论文的式 4 要求残差分支在初始化时为零，让每个块一开始就是恒等映射。加上 `identity_init`
后，8 块网络在初始化时的输出与"跳过所有块"完全一致（差 **0.0**，两个变体都是）。

这不只是一个技巧：它让 64 块的网络从"等价于浅网络"开始训练，而不是从一个随机深的函数开始。

## 差异归因

**1. 这测的是相对传输比例，不是可训练性——必须说清。** 两个变体的**绝对范数都在爆炸**
（pre 从 4.1 涨到 3.9e14）。原因是本切片**没有实现 BatchNorm**，残差流会无界累积。
论文的 1001 层结论依赖 BN；**本切片不能、也没有声称"恒等映射让 1000 层可训练"**。

所以能成立的只有一句：在同样没有归一化的条件下，恒等 shortcut 让梯度回传的**比例**
高出三个数量级（32 → 64 block 时 1461× → 11214×）。这是必要条件，不是充分条件。

**2. 判断不了"哪个更好"。** 梯度更大不等于训练更好——爆炸的梯度一样训不动。要回答
"v2 是否更好"，必须加上 BN 与真实训练，那是另一个量级的工作。

**3. 没有复现 4.62% / 200 层 ImageNet。** 没有数据集，也没有 BN。

**4. 与前三个"未复现"切片的区别**：这里既不是复现也不是反驳，而是**只验证了机制的一半**
（信号确实传得过去），并明确没验证另一半（传过去之后能不能训好）。这种"半复现"最容易
被写成"复现"，所以单独列出来。

## 跑

```bash
cd python/12_resnet_v2 && python3 test_resnet_v2.py    # 0.2 秒
```

## 下一步

- 若要补上另一半，需要 BatchNorm 的前向/反向 + 真实训练，属于 `11_resnet` 的延伸。
