# reconcile · 对账

把 Ilya Sutskever 给 John Carmack 的那份「约 30 篇」深度学习必读清单，从原始论文
重新实现一遍。**MIT 授权**。

在线浏览：<https://cordislabs.github.io/reconcile/>

## 形式

每篇一个目录，自带实现、测试和一份 `README.md`。那份 README 固定三段：

```
论文报告：<原文说了什么>
本仓实测：<测到了什么数字>
差异归因：<差在哪，为什么>
```

测试不用 pytest——每个 `test_*.py` / `cargo test` 自己就是入口。每篇至少有一个**不依赖训练**
的正确性来源：解析解、精确等价、小规模暴力枚举，或有限差分梯度校验。

## 结构

```
SYLLABUS.md            29 篇清单，每篇的可验证指标
python/                23 篇，Python + NumPy
  _shared/             梯度校验、共享语料
  07_char_rnn/         字符级 RNN + BPTT
  08_lstm/             LSTM + 梯度回收测量
  09_rnn_reg/          variational dropout 消融
  10_alexnet/          小卷积网 + ReLU/tanh 对比
  11_resnet/           退化现象与残差连接
  12_resnet_v2/        pre-activation 的梯度回传
  13_dilated/          膨胀卷积与感受野
  14_cs231n/           卷积/池化算子与 VGG16 计数
  15_bahdanau/         加性注意力
  16_attention/        多头注意力块 + 手写反向传播
  17_annotated_transformer/  完整编码-解码 Transformer
  18_order_matters/    集合编码器的置换不变性
  19_pointer/          指针网络 + 排序外推
  20_relational/       关系模块
  21_relational_rnn/   槽间注意力（MHDPA）
  22_mpnn/             消息传递
  23_ntm/              可微分记忆的寻址机制
  24_ctc/              CTC 损失与暴力枚举对照
  25_vlae/             后验坍缩测量
  26_gpipe/            微批流水线的等价性与峰值内存
  27_scaling_laws/     幂律拟合与算力最优分配
  28_ppo/              裁剪目标 + CartPole
  29_rlhf/             Bradley–Terry 偏好模型
rust/                  6 篇，Rust，零依赖
  m01_mdl/             MDL 二部码 + 噪声权重
  m02_mdl_tutorial/    NML 后悔值与 0.5 log n 渐近
  m03_complexodynamics/  log t 上界论证（依赖 m04）
  m04_coffee/          咖啡自动机
  m05_huffman/         Huffman + 熵界断言
  m06_msi/             Υ 智能度量（有限环境类）
```

## 怎么跑

```bash
make test                      # 各切片测试 + 目录与 SYLLABUS 的一致性

cd python/16_attention && python3 test_attention.py
cd rust && cargo test
```

训练轨只需 numpy；模拟轨只需 cargo。

## 篇目

| # | 切片 | 轨 | 状态 |
|---|---|---|---|
| 01 | `rust/m01_mdl` | Rust | ✅ MDL 二部码 + 噪声权重 |
| 02 | `rust/m02_mdl_tutorial` | Rust | ✅ NML 后悔值 |
| 03 | `rust/m03_complexodynamics` | Rust | ✅ log t 上界 |
| 04 | `rust/m04_coffee` | Rust | ✅ 咖啡自动机 |
| 05 | `rust/m05_huffman` | Rust | ✅ Huffman |
| 06 | `rust/m06_msi` | Rust | ✅ Υ 智能度量 |
| 07 | `python/07_char_rnn` | Python | ✅ 字符级 RNN |
| 08 | `python/08_lstm` | Python | ✅ LSTM + 梯度回收 |
| 09 | `python/09_rnn_reg` | Python | ✅ variational dropout |
| 10 | `python/10_alexnet` | Python | ✅ AlexNet / ReLU |
| 11 | `python/11_resnet` | Python | ✅ ResNet 退化现象 |
| 12 | `python/12_resnet_v2` | Python | ✅ pre-activation |
| 13 | `python/13_dilated` | Python | ✅ 空洞卷积 |
| 14 | `python/14_cs231n` | Python | ✅ CS231n 算子 |
| 15 | `python/15_bahdanau` | Python | ✅ Bahdanau 注意力 |
| 16 | `python/16_attention` | Python | ✅ 多头注意力 |
| 17 | `python/17_annotated_transformer` | Python | ✅ 完整 Transformer |
| 18 | `python/18_order_matters` | Python | ✅ 集合编码器 |
| 19 | `python/19_pointer` | Python | ✅ 指针网络 |
| 20 | `python/20_relational` | Python | ✅ 关系模块 |
| 21 | `python/21_relational_rnn` | Python | ✅ 槽间注意力 |
| 22 | `python/22_mpnn` | Python | ✅ 消息传递 |
| 23 | `python/23_ntm` | Python | ✅ NTM 寻址 |
| 24 | `python/24_ctc` | Python | ✅ CTC |
| 25 | `python/25_vlae` | Python | ✅ 后验坍缩 |
| 26 | `python/26_gpipe` | Python | ✅ 微批流水线 |
| 27 | `python/27_scaling_laws` | Python | ✅ Scaling Laws |
| 28 | `python/28_ppo` | Python | ✅ PPO |
| 29 | `python/29_rlhf` | Python | ✅ Bradley–Terry |

## 不包含

- **论文 PDF**：版权在各出版方，仓库只放链接。
- **数据集**：ImageNet / QM9 等有各自许可，仓库只放下载脚本或合成数据。
- **GPU 训练**：所有训练都在 CPU 上，秒级到分钟级。

## 授权

MIT，见 `LICENSE`。论文本身仍归原作者与出版方所有。
