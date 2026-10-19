# 04 · Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton（Aaronson, Carroll, Ouellette, 2014）

咖啡杯的二维格子模型：**熵（细数组的压缩长度）单调上升，而复杂度（粗粒化后压缩长度）先升后降**。

## 论文报告

模型（§3.1 原文，interacting model）：

> at each time step, one pair of horizontally or vertically adjacent, differing particles
> is selected, and the particles' positions are swapped.

初始状态（§3 原文）：

> The automaton begins in a state in which the top half of the cells are filled with ones,
> and the bottom half is filled with zeros.

两个度量（§4-§5 原文）：

> The estimated complexity of the state, K(S), is the file size of the thresholded,
> coarse-grained array after compression. Analogously, the estimated entropy of the
> automaton state is the compressed file size of the fine-grained array.

粗粒化的做法（§5.1）：每个格子取**以它为中心的 g×g 方块的均值**（尺寸不变），再把均值
按 0.4 / 0.6 阈值分成三档（咖啡 / 混合 / 奶油），最后压缩。

## 本仓实测

n = 64，grain g = 12，1000 万次交换（0.4 秒）。

| 步数 | 熵（细数组） | 复杂度（粗数组） |
|---|---|---|
| 0 | 90 | **111** |
| 1,333,333 | ~1490 | **537（峰值）** |
| 10,000,000 | 1516 | **110** |

- **复杂度：111 → 537 → 110**，峰值在 133 万步（内部，不在端点），回到起点量级。
- **熵：90 → ~1516**，涨约 17 倍；相邻采样点之间最大单次回落 **2.1%**（压缩估计器的噪声）。
- **粒子数守恒**：暗格比例全程严格 0.5000。
- 曲线确定性：同种子两次运行逐位相同。

四条断言全部通过（0.4 秒）。

## 第二个压缩器：曲线形状是不是压缩器的产物

论文用 gzip，本切片用自写 LZ77。为了排除"曲线是压缩器造成的"，加了一个 **order-2 上下文
模型**（按前两个符号的条件熵计长度），它与 LZ77 的算法完全不同：

| 压缩器 | 峰值步数 | 起点 | 峰值 | 终点 |
|---|---|---|---|---|
| LZ77 | 1,333,333 | 111 | **537** | 110 |
| order-2 | 1,333,333 | 2.6 | **211.4** | 20.7 |

**"先升后降"两个压缩器都成立**（峰值在同一个时间点）。但**"降回起点"只有 LZ77 成立**：
order-2 的终点（20.7）比起点（2.6）高很多，因为终态的粗粒化数组含有局部密度涨落，
上下文模型会把它算成信息量。

**结论**：这个模型里的"复杂度先升后降"对压缩器选择**稳健**；"最终回到起点"**不稳健**，
它依赖压缩器对残留涨落的敏感度。

## 差异归因

**1. 压缩器是我自己写的 LZ77，不是 gzip；另有一个 order-2 模型做对照。** 论文用 gzip/bzip。绝对值不可比，曲线形状可比——
这也是论文自己的态度（他们论证过 gzip 这类工具足够）。

**2. grain size 是我选的 12，论文说它是"experimentally selected"。** 这个选择很关键：
g=6 时局部均值标准差 0.083，即使完全混合也有大量格子越过阈值，**曲线降不回去**（实测只降到
879）。g=12 时标准差 0.042，降回 110。所以"先升后降"在这个模型里**依赖粗粒化的尺度**——
这不是还原论的细节，而是这个度量的性质。

**3. 规模比论文小。** n=64、1000 万步；论文的模拟更大、更久。绝对值不可比。

**4. 熵的"单调"是压缩估计意义下的。** 单点最大回落 2.1%，来自 LZ77 的贪心匹配对局部结构的
敏感性。论文也承认压缩只是近似（§4 有对 OSCR 失败的讨论）。

**5. 结论依赖定义。** 若把"熵"理解成细网格的 Shannon 熵、把"复杂度"理解成下采样后的块均值，
两条曲线的方向都会反过来。本切片用的是论文的操作定义：熵 = 细数组的压缩长度，复杂度 = 粗粒化
阈值化数组的压缩长度。

## 跑

```bash
cd rust && cargo test -p m04_coffee && cargo run -p m04_coffee --example report_m04_coffee
```

## 下一步

- `03_complexodynamics`：同一组作者的理论文章，讲为什么熵增与复杂度先升后降不矛盾。
