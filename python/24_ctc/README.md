# 24 · Deep Speech 2（Amodei et al., 2015）— CTC

本切片实现的是 CTC 损失：把帧序列对齐到较短的标签序列。

> 注：CTC 本身出自 Graves et al. (2006)，Deep Speech 2 是它的规模化应用。切片实现的是
> 损失函数，不是 Deep Speech 2 的语音系统。

## 论文报告

摘要原文（说明这篇论文的贡献在系统规模）：

> We show that an end-to-end deep learning approach can be used to recognize either English
> or Mandarin Chinese speech—two vastly different languages. Because it replaces entire
> pipelines of hand-engineered components with neural networks, end-to-end learning allows
> us to handle a diverse variety of speech including noisy environments, accents and
> different languages.

> Key to our approach is our application of HPC techniques, resulting in a 7x speedup over
> our previous system.

CTC 的目标：`P(l|x) = Σ_{π ∈ B^{-1}(l)} Π_t p(π_t | x)`，即对所有能塌缩成 `l` 的帧序列
求和。

## 本仓实测

| 检验 | 实测 |
|---|---|
| 动态规划 vs 暴力枚举（6 个目标，含重复与空） | 差 ≤ **8.9e-16** |
| **CTC 梯度 vs 有限差分**（4 种情形，含空目标与重复） | max_abs **8.1e-10** |
| **beam search vs 贪心**：序列概率 | beam **不低于**贪心 |
| 置信模型解码回原序列 | `[1,2]` ✅ |
| 所有输出序列的概率之和（T=5，V=3，全部 243 条路径） | **1.000000000000** |
| DP 对全部 **25** 个可达输出序列的匹配 | 最大差 **8.9e-16** |
| `[1,1]` 从 T=2 帧产生 | 不可能（loss = 1e+30） |
| `[1,1]` 从 T=3 帧产生 | 可能 |
| 目标长于序列 | 不可能 |

分区和等于 1 这一条是最强的检查：它一次性验证了塌缩规则在全部 243 条路径上的正确性，
并且逐个验证 DP 与枚举质量一致。

## 差异归因

**1. 只实现了损失，没有语音系统。** 论文的核心是端到端声学模型 + HPC 工程（7× 加速、
Batch Dispatch）。本切片没有音频、没有模型、没有解码。

**2. 解码实现了，但只是前缀 beam search。** 贪心与 beam 都有；beam 保证不差于贪心（搜索空间包含它），
在合成 logits 上验证。没有测真实语音的解码误差。

**3. 梯度实现了，且对有限差分校验过（8.1e-10）。** 过程中修了两个真 bug：beta 递推的"保持状态"
分支漏了 `y[t+1]`；softmax 的雅可比写成 `y*(g-Σy·g)`，正确的是 `g - y·Σg`。

**4. 任务规模极小**：T=5、词表 3 —— 这正是为了能暴力枚举。真实语音是 T=数千帧。

**5. 第一版测试写错过。** 我原来写"帧数增加则 loss 下降"，但每组的随机帧不同，本来就不
单调。换成了分区和 = 1 这条真正成立的性质。**测试也要有自己的判据**，否则它只是在断言
一个我编出来的规律。

## 跑

```bash
cd python/24_ctc && python3 test_ctc.py    # 0.1 秒
```
