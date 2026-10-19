# 28 · Proximal Policy Optimization（Schulman, Wolski, Dhariwal, Radford, Klimov, 2017）

裁剪代理目标 + CartPole 上的训练。

## 论文报告

摘要原文：

> We propose a new family of policy gradient methods for reinforcement learning, which
> alternate between sampling data through interaction with the environment, and optimizing
> a "surrogate" objective function using stochastic gradient ascent.

> … we propose a novel objective function that enables multiple epochs of minibatch
> updates. The new methods, which we call proximal policy optimization (PPO), have some of
> the benefits of trust region policy optimization (TRPO), but they are much simpler to
> implement …

目标函数（式 7）：

```
L^CLIP(θ) = E[ min( r_t(θ)·Â_t , clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]
```

## 本仓实测

**裁剪的梯度语义（精确，不依赖训练）**：

| 情形 | 梯度 dL/dlogp | 实测 |
|---|---|---|
| r > 1.2，A > 0 | 恰为 **0** | 0.0 |
| r > 1.2，A < 0 | −rA = 1.5 | 1.5 |
| r < 0.8，A < 0 | 恰为 **0** | 0.0 |
| r < 0.8，A > 0 | −rA = −0.5 | −0.5 |
| 区间内（0.85 / 1.0 / 1.15） | −rA | 一致 |

哪一支获胜取决于优势的符号——两个符号都验了，否则"外侧梯度为 0"这句话是错的。

**训练**：CartPole，60 轮迭代，3.8 秒。贪心评估的平均回报 **499.8–500.0**（环境上限 500，
五个种子）；采样评估在 **468–500** 之间波动——那是动作采样的方差，不是策略质量的差异。

## GAE 与更难的环境（GPU）

给 PPO 加了广义优势估计（GAE），并在 CartPole 和一个更难变体（杆长加倍）上比较
"达到评估 475 分需要多少环境步"。每个配置跑 5 个种子：

| 环境 | 无 GAE（中位步数） | 有 GAE（中位步数） |
|---|---|---|
| CartPole | 2,830 | 16,882 |
| CartPole（杆长 1.0） | 15,922 | 15,218 |

**这个对比没有结论。** 单次运行的步数从 **489 到 29,268**，方差远大于两组之间的差。
5 个种子不足以区分，而"步数到阈值"这个统计量本身很粗糙（评估只有 10 个 episode）。

GAE 的边界性质倒是可以精确验证（SPL 见测试）：λ=0 时等于 TD 误差，λ=1 时等于蒙特卡洛优势。

## MuJoCo（GPU）

论文的基准是连续控制的 MuJoCo，不是 CartPole。`gpu/mujoco_ppo.py` 是一个连续动作的 PPO
（高斯策略 + GAE + 同样被裁剪的目标），跑 HalfCheetah-v4，50 万环境步、14 分 36 秒：

| 迭代 | 平均回报 |
|---|---|
| 144 | 519 |
| 168 | 722 |
| 192 | 1102 |
| 216 | 1215 |
| 240 | **1280** |

随机初始化的 HalfCheetah 回报约 −300，所以这是一条真实的学习曲线。但**离论文的数字还有距离**
（论文的 HalfCheetah 在数百万步内到 2000 以上）：没有做观测归一化、没有调 lr 和 rollout 长度、
只跑了一个种子。

## 差异归因

**1. CartPole 不在论文里——所以补了 MuJoCo。** 论文的基准是 MuJoCo 运动任务与 Atari。CartPole 是教学环境，
我用它是因为 CPU 上几秒能跑完——所以"500 分"**不能**与论文的任何数字对照。MuJoCo 那一节补上了
这个对照，但只跑到论文级数的一半左右。

**2. GAE 实现了，但没测出收益。** λ=0/1 的边界性质精确成立；在 CartPole 上，5 个种子的
方差（489–29,268 步）远大于 GAE 与折扣回报之间的差。这是本切片最大的算法简化。

**3. 评估用贪心而非采样。** 采样评估多 30 个 episode 的方差可达 30 分（同配置 468.4 与
500.0 并存），在 CI 上会把一个固定的 475 阈值判成失败。判据用贪心，文档记录两种评估的数字。

**4. 没有并行环境、没有 entropy bonus、单一种子。** 论文用多个并行 worker 收集数据，并在
目标里加熵奖励；这里单环境、无熵项、只跑了一个种子。CartPole 训练对种子敏感，"500 分"
是一个种子下的结果。

**5. 裁剪那条判据的价值在于它会抓错。** 如果 `min` 写成了 `max`，或者 clip 的上下界写反，
外侧梯度会非零——测试立刻失败。这是本切片唯一不依赖训练的判据，也是最可靠的。

## 跑

```bash
cd python/28_ppo && python3 test_ppo.py    # 3.9 秒
```

## 下一步

- `29_rlhf`：用 PPO 做人类偏好对齐，接上 28。
