# 05 · Kolmogorov Complexity and Algorithmic Randomness（Shen, Uspensky, Vereshchagin）

Huffman 编码，**零依赖**。按"最优前缀码"的定义实现，不是照抄任何库。

## 论文报告

Shannon 信源编码定理：对无记忆信源，Huffman 码的平均码长满足 **H ≤ L < H + 1**；
Kraft 不等式 Σ 2^(−ℓᵢ) = 1 对满二叉树恰好取等。

## 本仓实测

`cargo run -p m05_huffman --example report_m05_huffman` 的输出：

| 语料 | 符号数 | 熵 H | Huffman L | L−H | vs 定长 8 bit |
|---|---|---|---|---|---|
| 英文样例（205 字节） | 28 | 4.2984 | **4.3366** | 0.0382 | **54.21%** |
| 偏斜 5 符号（a 占 90%） | 5 | 0.6338 | **1.1700** | 0.5362 | vs 定长 3 bit |

6 个测试全部通过：Kraft 取等（误差 < 1e-12）、前缀无关性（逐对穷举）、
H ≤ L < H+1、偏斜源优于定长、编解码往返一致、单符号字母表（码长 0 = 熵 0）。

## 差异归因

- 英文样例上 L−H 只有 **0.038 bit**，远好于 +1 的上界——因为符号少（28 个）且
  分布相对平缓，Huffman 恰好接近最优。**样本越小，这个"好消息"越不可信**：
  上界是渐近结论，不是小样本保证。
- 偏斜源上 L−H = 0.536，接近 +1 上界——这是 Huffman 的已知弱点：它只能分配整数
  码长，对概率极不均衡的源浪费大。补上算术编码（达到 ⌈−log₂p⌉）才完整，属于本切片
  的下一步。

## 跑

```bash
cd rust && cargo test && cargo run -p m05_huffman --example report_m05_huffman
```

## 下一步

算术编码（区间编码）与 NCD 聚类——指标里的后两条还没做。
