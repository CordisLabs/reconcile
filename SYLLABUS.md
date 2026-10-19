# 29 篇清单与可验证指标

> 源头：Ilya Sutskever 给 John Carmack 的推荐，原话 "If you really learn all of these,
> you'll know 90% of what matters today"，由 @keshavchan 于 2024 年公开推文转述
> （<https://twitter.com/keshavchan/status/1787861946173186062>）。
> 这份转录清点下来是 **27 条**，不是 30——原文写的就是 "approx 30"。
>
> 本仓库在 27 条之外补两篇：**PPO**、**RLHF**。理由：2017 年之后，这两篇是把前面
> 27 篇接进"今天的模型"的唯一通路。补什么、为什么补，写在这里供反对。

## 两条轨的划分标准

不是按难度分，是按**是否需要反向传播**分。需要梯度的 23 篇用 Python + NumPy；
纯模拟、压缩、信息论的 6 篇用零依赖 Rust——那 6 篇没有 backprop，Rust 的快、
确定性与"十年后仍能编译"是真收益。

## 指标的含义

每篇只认一个**数字**。格式固定三行：

```
论文报告：<原文说了什么>
本仓实测：<我们测到什么>
差异归因：<差在哪，为什么>
```

没有第三行的仓库就是"能跑但不知道对不对"，那正是现有复现仓库的通病。

---

## A. 信息、压缩与复杂性（Rust 轨 · 6 篇）

| # | 论文 | 年 | 可验证指标 |
|---|---|---|---|
| 01 | Keeping Neural Networks Simple by Minimizing the Description Length of the Weights — Hinton & van Camp <https://www.cs.toronto.edu/~hinton/absps/colt93.pdf> | 1993 | ✅ 二部码长在 σ 网格上有内点极小；**固定模型时 MDL 选的 σ 比测试最优偏小**（代价 1.02–1.49×，随先验尺度）；跨 degree 选择代价 1.116×；噪声权重部分：解析梯度 <1e-6，噪声水平被学成内点最优（总码长 193→4.85 nats） |
| 02 | A Tutorial Introduction to the Minimum Description Length Principle — Grünwald <https://arxiv.org/abs/math/0406077> | 2004 | ✅ NML 后悔值：R(1)=1.0000 … R(1e5)=**8.6330**；斜率 **0.5**（误差 <0.02）；Kraft 和 = **1.000000000000000**；偏移收敛到 **0.5·log₂(π/2)=0.3258**（实测 0.3282）。只做 NML，未做 two-part / prequential / AIC-BIC 对比 |
| 03 | The First Law of Complexodynamics — Aaronson | 2011 | ✅ 上界论证：shortcut 74→88 bits vs 状态 1272→12128 bits。**complextropy 扫描**（6 组 grain × 长度）：**上升稳健**（−12 → 450–820），**下降不稳**（grain=6 降 25%、grain=14 **降 0%**）——**部分支持**，依赖粗粒化尺度 |
| 04 | Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton — Aaronson et al. <https://arxiv.org/abs/1405.6903> | 2014 | ✅ 按论文定义复现：熵 90→1516；复杂度 111→537→110。**双压缩器对照**：LZ77 与 order-2 的峰值都在 **1,333,333 步**（537 / 211.4 字节），所以「先升后降」**对压缩器稳健**；但「降回起点」只有 LZ77 成立（order-2 终点 20.7 vs 起点 2.6） |
| 05 | Kolmogorov Complexity and Algorithmic Randomness — Shen, Uspensky, Vereshchagin <https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf> | — | ✅ Huffman：英文样例 H=4.2984 / L=4.3366（54.21% of 定长 8 bit），H ≤ L < H+1、Kraft 取等 <1e-12、前缀无关、往返一致；偏斜源 L=1.1700 vs 定长 3 bit。**未做**：算术编码与 NCD（指标里的后两条）|
| 06 | Machine Super Intelligence — Legg <https://www.vetta.org/documents/Machine_Super_Intelligence.pdf> | 2008 | ✅ Υ 有界；三档 agent 排序成立（随机 **0.5024** / 最优固定 **0.5145** / oracle **0.9412**）；归一化 Υ 对编码平移严格不变（c=0…16，差 <1e-12），未归一化按 2^(−c) 缩放。**K 不可计算**，用有限环境类代替；原 PDF 取不到，公式未逐字核对 |

原始链接：<https://www.cs.toronto.edu/~hinton/absps/colt93.pdf> ·
<https://arxiv.org/abs/math/0406077> · <https://scottaaronson.blog/?p=762> ·
<https://arxiv.org/abs/1405.6903> ·
<https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf> ·
<https://www.vetta.org/documents/Machine_Super_Intelligence.pdf>

---

## B. 序列模型的基础（Python 轨 · 3 篇）

| # | 论文 | 年 | 可验证指标 |
|---|---|---|---|
| 07 | The Unreasonable Effectiveness of RNNs — Karpathy <https://karpathy.github.io/2015/05/21/rnn-effectiveness/> | 2015 | ✅ 字符级 RNN + BPTT：val **2.952** bits/char（order-0 **4.495**），梯度 5.9e-7。**n-gram KL 曲线**：unigram 一开始就对齐（0.052），trigram 从 **8.13 → 5.26** |
| 08 | Understanding LSTM Networks — Olah | 2015 | ✅ 梯度保留：T=20 **412×**、T=40 **≈10⁶×**（精确测量）。**copy 预算扫描**：T≤20 时 vanilla 在每个预算下更好（T=20/8000 步：0.876 vs 0.470）；**T=50 两者都在随机水平**（0.102）——长延迟区间**我的程序不够格，该主张未检验** |
| 09 | Recurrent Neural Network Regularization — Zaremba et al. | 2014 | ✅ 无 dropout 的留出损失在最低点后持续上升，**退化随预算增长**（1200/2500/5000 步：0.249 → 1.233 → **3.130**）；有 dropout 基本贴住最低点（≤0.21）。5000 步时差 **15 倍** |

链接：<https://karpathy.github.io/2015/05/21/rnn-effectiveness/> ·
<https://colah.github.io/posts/2015-08-Understanding-LSTMs/> ·
<https://arxiv.org/abs/1409.2329>

---

## C. 视觉与深度（Python 轨 · 5 篇）

| # | 论文 | 年 | 可验证指标 |
|---|---|---|---|
| 10 | ImageNet Classification with Deep CNNs (AlexNet) — Krizhevsky et al. | 2012 | ✅ 反向 4.2e-10；ReLU 在 0 处解析/数值 0.0 / 0.5。**论文设定重做（SGD + 四层卷积 + 每网络调 lr）**：ReLU **0.6935** vs tanh 0.6270（+6.7 点），达到 loss 1.0 快 **1.27×**——**方向复现、幅度不是 6×** |
| 11 | Deep Residual Learning (ResNet) — He et al. <https://arxiv.org/abs/1512.03385> | 2015 | ✅ **退化现象复现**：合成 24 块 MLP（0.100 vs 0.993）；**CIFAR-10 上 10/20/40 层**：plain 0.6587 → 0.6636 → **0.5599**（40 层掉 10 个点），residual 0.7003 → 0.7359 → **0.7364**（单调不降） |
| 12 | Identity Mappings in Deep Residual Networks (ResNet v2) — He et al. <https://arxiv.org/abs/1603.05027> | 2016 | ✅ pre/post 对比（1461×）；加 BN 后 64 块 dL/dx 3.86e14→**5.25e01**；**identity 初始化（式 4）**：8 块在初始化时与跳过全部块**完全一致**（差 0.0） |
| 13 | Multi-Scale Context Aggregation by Dilated Convolutions — Yu & Koltun <https://arxiv.org/abs/1511.07122> | 2015 | ✅ 膨胀卷积 ≡ 零插入核的普通卷积（max\|Δ\| = **0.0**）；[1,2,4] → **15×15**，7 层 `1,1,2,4,8,16,1` → **67×67**（与论文一致，故 padding 33）；覆盖 225/225 无洞；参数 63 vs 单个 67×67 的 4489；分辨率不变 |
| 14 | CS231n: Convolutional Neural Networks for Visual Recognition — Karpathy, Li <https://cs231n.stanford.edu/> | — | ✅ im2col 卷积 == 朴素卷积（max\|Δ\| ≤ **5.3e-15**，非逐位相等）；输出尺寸公式 4 组一致；max-pool 与反向路由逐位正确；**VGG16 参数量 = 138,357,544**（依赖 7×7×512 / 1000 类 / FC 带 bias 三个约定）|

链接：<https://papers.nips.cc/paper/4824> · <https://arxiv.org/abs/1512.03385> ·
<https://arxiv.org/abs/1603.05027> · <https://arxiv.org/abs/1511.07122> ·
<https://cs231n.github.io/>

---

## D. 注意力（Python 轨 · 4 篇）

| # | 论文 | 年 | 可验证指标 |
|---|---|---|---|
| 15 | Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. <https://arxiv.org/abs/1409.0473> | 2014 | ✅ 反转任务上精确匹配 **1.00**、对齐命中 **1.00**（长度 5）；注意力行和误差 1.1e-16；梯度 5.6e-10。**长度泛化有限**：只训 5 → 长度 6 归零；混合 3–6 → 7 尚可（0.97）、9 崩（0.02）。无 BLEU（toy 任务，且用 vanilla 单元而非 GRU）|
| 16 | Attention Is All You Need — Vaswani et al. <https://arxiv.org/abs/1706.03762> | 2017 | ✅ 多头注意力解析梯度 vs 有限差分 **4.0e-10**；置换等变性 **5.6e-17**；∂L/∂b_k ≡ 0（softmax 平移不变，实测 6.3e-17）。**未复现** BLEU（无 WMT14 / GPU，切片只到注意力块）|
| 17 | The Annotated Transformer — Rush | 2018 | ✅ 梯度 4.9e-10；因果掩码 Δ=0.0；copy 1.00。**label smoothing 三部分**：伤 ppl ✅（1.0008→1.0981，贴推导下限）、**提升泛化 ✅**（训练长 8、测长 14：ppl 45.1→5.6）、提升准确率 ✗（两边都 0，无长度外推） |
| 18 | Order Matters: Sequence to Sequence for Sets — Vinyals et al. <https://arxiv.org/abs/1511.06391> | 2015 | ✅ **结构**：集合编码器池化上下文在置换下差 **2.8e-17**，LSTM 编码器差 0.12–0.19。**效果**：排序 n=6 **1.000**、n=8 **1.000**（LSTM 0.920 / 0.830）。梯度 1.3e-10。**只做了输入侧不变性**，论文的输出侧 order loss 未做 |

链接：<https://arxiv.org/abs/1409.0473> · <https://arxiv.org/abs/1706.03762> ·
<https://nlp.seas.harvard.edu/annotated-transformer/> · <https://arxiv.org/abs/1511.06391>

---

## E. 结构化归纳偏置（Python 轨 · 6 篇）

| # | 论文 | 年 | 可验证指标 |
|---|---|---|---|
| 19 | Pointer Networks — Vinyals et al. | 2015 | ✅ 排序 n=6 **0.940**，外推 n=10 0.75。**TSP（2D）**：训练量 2 万→48 万实例，精确匹配 **0.025 → 0.450**，路径 **1.357 → 1.053×最优**——早先的「没学会」是训练量不足，不是论文错 |
| 20 | A Simple Neural Network Module for Relational Reasoning — Santoro et al. | 2017 | ✅ 置换不变 4.4e-15；计数任务 0.985。**平衡标签后的对照**：同一份物体特征上 MLP **0.643** vs RN **0.803**（+16 点）；图像上 CNN 0.819 vs CNN+RN 0.853（+3.4 点，全局池化能靠数颜色猜对大半）——**主张成立，幅度取决于问题可被统计回答的比例** |
| 21 | Relational Recurrent Neural Networks — Santoro et al. | 2018 | ✅ 槽间 MHDPA：行和 1.1e-16、等变 2.2e-16、梯度 6.4e-10。**与 LSTM 对比**：L=20 时 LSTM 0.316 / 一次性 **1.000**；L=50 时 LSTM 0.323 / 一次性 **1.000**，但**三种逐时间步实现都是 0.323**——一次性注意保持、逐时间步稀释 |
| 22 | Neural Message Passing for Quantum Chemistry — Gilmer et al. <https://arxiv.org/abs/1704.01212> | 2017 | ✅ 置换不变 5.6e-17、等变 4.4e-16、梯度 9.4e-11；合成图回归 MAE 0.563 vs 1.970。**边特征 + 消融**：无边形特征 **2.567**，有边 **1.171**（降到 45%）。没有 QM9 |
| 23 | Neural Turing Machines — Graves et al. | 2014 | ✅ 寻址算子精确。**copy 任务**：课程 65,000 步后训练长度 **97.58%**、修复 content addressing 门控后 **95.39%**，**两版外推 15/20 都是 0%**——不是训练量，是长度绑定策略；归因缩小到槽数/写头 allocation/训练分布 |
| 24 | Deep Speech 2 — Amodei et al. <https://arxiv.org/abs/1512.02595> | 2015 | ✅ CTC：DP 与暴力枚举差 ≤8.9e-16；全部 243 条路径塌缩质量 = 1.000000000000；**梯度 vs 有限差分 8.1e-10**（4 种情形）；**贪心 + 前缀 beam 解码**，beam 不差于贪心、无 blank。未测真实语音解码 |

链接：<https://arxiv.org/abs/1506.03134> · <https://arxiv.org/abs/1706.01427> ·
<https://arxiv.org/abs/1806.01822> · <https://arxiv.org/abs/1704.01212> ·
<https://arxiv.org/abs/1410.5401> · <https://arxiv.org/abs/1512.02595>

---

## F. 生成与规模化（Python 轨 · 3 篇）

| # | 论文 | 年 | 可验证指标 |
|---|---|---|---|
| 25 | Variational Lossy Autoencoder — Chen et al. | 2016 | ✅ 后验坍缩：β 0.01→10 时 KL 10.242→0.102、活跃维度 8→4。**VLAE 两部件**（MNIST，16 维隐变量，2 epoch）：独立解码器 0.2350、自回归解码器 0.1719、**两者叠加 0.0945**（降 60%） |
| 26 | GPipe — Huang et al. | 2019 | ✅ 前向逐位相同；峰值激活 6144→1184（5.19×）；梯度 vs 有限差分 9.2e-10。**重计算**：不存激活、反向重放，梯度**完全相同**（0.0），存储量与深度无关（K=16 省 **8×**，代价 O(K²) 重放） |
| 27 | Scaling Laws for Neural Language Models — Kaplan et al. <https://arxiv.org/abs/2001.08361> | 2020 | ✅ 8/10 个规模的真实曲线，数据充足子集 α=**0.0778**（r²=0.993），论文 0.076。**L(N,D) 联合拟合**：12 个点（4 规模 × 3 预算），**L = 14.90·N^−0.234 + 2044.9·D^−0.533 + 0.833**，r²=0.970；指数比论文大 2–5 倍（网格只有 200 万–2000 万 token） |

链接：<https://arxiv.org/abs/1611.02731> · <https://arxiv.org/abs/1811.06965> ·
<https://arxiv.org/abs/2001.08361>

---

## G. 强化学习与对齐（Python 轨 · 2 篇 · 本仓补充）

| # | 论文 | 年 | 可验证指标 |
|---|---|---|---|
| 28 | Proximal Policy Optimization — Schulman et al. <https://arxiv.org/abs/1707.06347> | 2017 | ✅ 裁剪梯度精确（外侧 0、内侧 −rA）；GAE λ=0/1 边界精确。CartPole 贪心评估 499.8–500.0。**MuJoCo HalfCheetah**：50 万步 / 14 分 36 秒，回报 519 → **1280**（随机约 −300），但不到论文的 2000+ |
| 29 | Learning to Summarize from Human Feedback (RLHF) — Stiennon et al. | 2020 | ✅ Bradley-Terry：隐藏奖励恢复 r=0.99。**真实人类偏好（HH-RLHF）**：哈希词袋 + 线性奖励模型，留出集一致率 **0.6033**（随机 0.5）——偏好可学；合成偏好那半仍保留 |

链接：<https://arxiv.org/abs/1707.06347> · <https://arxiv.org/abs/1706.03741>

---

## 附录：不在清单内、但值得单独加的

| 论文 | 为什么不在 | 建议 |
|---|---|---|
| Dropout — Srivastava et al. 2014 | 不在 Ilya 原始 27 条里（30u30 把它列进 Day 12） | 内容确实基础，实现成本低于 1 小时，可作附加篇 |
| BERT / GPT-2 / GPT-3 / Chinchilla | 原清单成文于 2020 年之前 | 属于"清单之后"的演进，另开一卷 |

> 这个表是**故意留的**：把"原清单"和"我们加的"分开写，读者才知道哪条是 Ilya 的、
> 哪条是我们的判断。混在一起是最容易招致的批评。
