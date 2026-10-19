# 06 · Machine Super Intelligence（Shane Legg, 2008）

Legg 的通用智能度量 Υ，在一个**有限环境类**上精确计算。

## 论文报告

原论文 PDF 从 `vetta.org` 取不下来（TLS 证书链不完整），USI 的副本需要登录，所以这里
引的是**机构库记录里的论文摘要**（[USI 记录](https://susi.usi.ch/usi/documents/317954)）：

> This thesis concerns the optimal behaviour of agents in unknown computable
> environments, also known as universal artificial intelligence.

> … In which settings the behaviour of universal agents converges to optimal, the way in
> which AIXI theory relates to the concept and definition of intelligence, the limitations
> that computable agents face …

度量本身的定义（通行表述，**未从原文逐字核对**）：

```
Υ(π) = Σ_μ 2^(−K(μ)) · V_μ^π
```

`K(μ)` 是环境的 Kolmogorov 复杂度，`V_μ^π` 是策略 π 在环境 μ 上的期望回报。

## 本仓实测

6 个确定性双臂老虎机，复杂度 1–7 比特，用 `2^(−K)` 加权。

| agent | 归一化 Υ |
|---|---|
| 随机（p = 0.5） | **0.5024** |
| 最优固定策略（p = 1.0） | **0.5145** |
| oracle（每个环境选最优臂） | **0.9412** |

**复杂度不变性**（把所有复杂度同时加 c）：

| c | 归一化 Υ（三档） | 未归一化 Υ（随机） |
|---|---|---|
| 0 | 0.5024 / 0.5145 / 0.9412 | 6.477e-1 |
| 1 | 同上 | 3.238e-1 |
| 3 | 同上 | 8.096e-2 |
| 8 | 同上 | 2.530e-3 |
| 16 | 同上 | 9.882e-6 |

归一化后三档数值在 c=0…16 下**完全不变**（差异 < 1e-12）；未归一化的值严格按 `2^(−c)`
缩放。这就是"换编码只改变尺度、不改变排序"的可算版本。

## 差异归因

**1. K 不可计算，所以这不是论文的 Υ。** 我用有限环境类 + 一个选定编码代替 Kolmogorov
复杂度——这正是论文承认的实践中不可避免的简化。所以本切片算的是 Υ 在有限类上的一个
实例，不是那个不可计算的通用量。

**2. 环境是确定性双动作老虎机。** `V` 精确可算，但没有部分可观测、没有随机转移、没有
序列决策——Legg 的框架里环境可以是有记忆的交互式环境，这里的差距很大。

**3. 公式没有逐字核对。** 原始 PDF 取不到（TLS），摘要引的是机构库记录。这一条必须写明，
否则读者会以为公式是逐字从论文抄的。

**4. 不变性成立，但很浅。** 把每个复杂度加同一个常数，只是让选定的先验整体缩放。真正的
复杂度不变性（换参考通用机）需要 K 本身，本切片没有触及。

**5. 随机与最优固定策略差距很小（0.5024 vs 0.5145），是环境类太贫乏，不是"智能差异小"。**
六个几乎对称的老虎机里，一个不看环境的策略本来就没什么可损失的。oracle 的 0.94 才是
"有信息"和"没信息"的真实差距。

## 跑

```bash
cd rust && cargo test -p m06_msi && cargo run -p m06_msi --example report_m06_msi
```

## 下一步


