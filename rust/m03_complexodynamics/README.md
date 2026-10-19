# 03 · The First Law of Complexodynamics（Aaronson, 2011）

博客文章，不是论文。本切片验证的是它的**上界论证**——这一条是算术，可以精确检查。

## 论文报告

文章要回答的问题（原文）：

> why does "complexity" or "interestingness" of physical systems seem to increase with
> time and then hit a maximum and decrease, in contrast to the entropy, which of course
> increases monotonically?

核心论证（原文）：

> if the system happens to be *deterministic*, then mathematically, its state can always be
> specified by giving (1) the initial state, and (2) the number of steps t it's been run
> for. The former takes a constant number of bits to specify (independent of t), while the
> latter takes log(t) bits. It follows that, if we use Kolmogorov complexity as our stand-in
> for entropy, then the entropy can increase at most *logarithmically* with t.

文章的猜想（原文）：

> I conjecture that the complextropy will satisfy the "First Law of Complexodynamics" …
> I don't yet know how to prove this conjecture.

## 本仓实测

用 `m04_coffee` 的咖啡自动机（workspace 路径依赖，不重复实现）。整个轨迹是种子的确定性
函数，所以状态可以由 `(seed, t)` **精确重建**：

| t | shortcut 长度（bits） | 压缩状态长度（bits） | 比值 |
|---|---|---|---|
| 1,000 | 74 | 1,272 | 17× |
| 10,000 | 78 | 2,424 | 31× |
| 100,000 | 81 | 5,520 | 68× |
| 1,000,000 | 84 | 11,544 | 137× |
| 10,000,000 | 88 | 12,128 | 138× |

- **shortcut 只涨了 14 bits（跨四个数量级）**，而压缩状态涨了近 10 倍。比值 17× → 138×。
- 1e9 步的步数描述只需 **30 bits**。
- 重建是可验证的：同一 `(seed, t)` 两次运行逐位相同；`t` 换成 `t+1000` 则不同。

三条断言全部通过（0.36 秒）。

## 猜想本身：complextropy 的形状

博客的猜想是"复杂度先升后降"。它提了一个可操作的量：**状态里未被粗粒化捕获的信息**，
可以用 `压缩(细) − 压缩(粗)` 估计。单次测量说明不了什么，所以扫了 grain 和演化长度
（48×48 网格，20 个采样点）：

| grain | 步数 | 起点 | 峰值 | 终点 | 从峰值下降 |
|---|---|---|---|---|---|
| 6 | 400 万 | −12 | 454 | 340 | **25%** |
| 10 | 400 万 | −12 | 757 | 633 | 16% |
| 14 | 400 万 | −15 | 818 | 818 | **0%** |
| 6 | 1200 万 | −12 | 467 | 342 | 27% |
| 10 | 1200 万 | −12 | 759 | 693 | 9% |
| 14 | 1200 万 | −15 | 817 | 813 | **0%** |

**结论：上升稳健，下降不稳。**

- **上升**：六种设定下都从 ~0 升到 450–820，方向一致。
- **下降**：grain=6 有 25–27%、grain=10 只有 9–16%、**grain=14 完全不平降**（终点就在峰值）。
- 峰值出现的位置也不稳定（第 1、11、15、16、20 个采样点都有）。

所以这条主张在本仓是**部分支持**：形状的"升"成立，"降"取决于粗粒化尺度，在较大的 grain 下
根本不出现。这和 `04` 的教训相同——**度量的结论依赖它的尺度参数**。

原文自己说"I don't yet know how to prove this conjecture"，这里做的是一个模型、一个度量、
三个 grain 的经验检查，不是证明。

## 差异归因

**1. 这篇是博客，提出的是猜想，不是结果。** 原文明确说 "I don't yet know how to prove
this conjecture"。本切片验证的是**上界论证**——"确定性系统的状态有 log t 的短描述"——
这一条成立，也就说明了为什么**必须**引入资源界：没有它，任何基于无界 Kolmogorov 复杂度的
度量都会被这个捷径压住，永远涨不起来。

**2. 猜想做了经验检查，只部分支持。** 要验证"complextropy 先升后降"需要资源受限复杂度，那是不可计算的。
`04_coffee` 用压缩代理做了一次经验性验证，但那不等于证明猜想——两篇的差别就在这。

**3. "重建"验证的是确定性，不是"存在短程序"这个哲学论断。** 我用同一份代码、同一个 RNG
重建，这只能说明轨迹由种子决定；真正的 `K(x)` 仍然不可计算。

**4. 本切片通过 path 依赖复用 `m04_coffee` 的自动机**，而不是重新实现一份。重复实现只会增加
出错面，不会多验证任何东西。

## 跑

```bash
cd rust && cargo test -p m03_complexodynamics && cargo run -p m03_complexodynamics --example report_m03_complexodynamics
```

## 下一步

- 20 关系推理（Relational Reasoning），回到训练轨。
