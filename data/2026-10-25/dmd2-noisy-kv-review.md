# DMD2 是否适合 4222＋Noisy KV reuse

值得做分项实验，但目前没有该模型的 DMD2 训练结果，不能承诺 FVD 会改善。Noisy KV reuse 决定历史怎样缓存，DMD2 改变蒸馏的监督与更新方式；二者可以结合，保持部署调用数不变。

## 原理与现有实现

DMD 用冻结 teacher 的 score 与可训练 fake score 之差指导学生。fake score 学习当前学生输出的分布；它不是 GAN 的真假分类器。DMD2 让 fake score 更新得更频繁，以减少它跟不上学生时的梯度偏差；加入真实数据的 GAN 监督；多步训练使用学生自己产生的中间状态。其原始 ODE 配对回归项被移除。[论文](https://arxiv.org/abs/2405.14867)

官方 ImageNet / SDXL 配方采用每次 student 更新对应5次 fake 更新；这不是所有模型的固定规则，SD1.5 示例也有10次。官方 UNet 的 GAN 分类分支接在 fake 网络特征上。移植到 Cosmos Transformer 需要设计视频特征及条件接口。[固定官方代码版本](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f) · [SDXL 配方](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh)

检查 CoStream `costream/objectives.py`、`costream/rollout.py` 和 `configs/stage3_422_noisy_matched_formal.yaml` 后：

| 项目 | 当前修正版 noisy | 值得验证的改动 |
|---|---|---|
| 分布匹配 | 已有冻结 real score、可训练 fake score、DMD | 保留 |
| 更新频率 | student:fake = 1:1 | 单独比较1:2与1:5；先看critic跟踪误差与稳定性 |
| 学生输入 | 已执行固定422…、学生自身历史、Noisy KV rollout | 保留训练和部署相同的缓存语义 |
| 梯度范围 | 每块所有denoise步反传；跨块历史detach | 与官方DMD2的抽取中间步训练并不完全相同，须另作消融 |
| 辅助目标 | AnyFlow forward / transition；历史噪声已匹配 | 这不是DMD旧版的离线teacher ODE回归，不能照搬“删回归”而直接删掉 |
| GAN | 没有真实视频真假判别器 | 可再加小权重条件视频判别监督；真实训练latent可供使用 |

## 建议实验顺序

1. 保持422…、Noisy KV策略、历史噪声匹配、score时间分布、学习率及forward权重，只改fake更新频率。增加按固定噪声时间统计的fake误差、DMD梯度幅度和student/fake实际更新数。此项应称为“DMD2式TTUR消融”，尚不是完整DMD2。
2. 若TTUR有收益，再加GAN项，比较同一Noisy KV轨迹。判别器需要结合首帧、动作和时间，不能只奖励单帧锐度；关注动作响应、运动、闪烁及后半段漂移。
3. 保持相同student更新数，并另报GPU小时；当前1600总步为800student＋800fake。若改1:5，保持800student需4800总更新，不能仍训1600步后直接比较。每步计算量不同，不能把总更新数比例当作时间比例。

GAN的真实样本只取训练split，不读172条validation；不需要预先生成一整套50步teacher轨迹。AnyFlow自身的训练数据需求仍然存在。

预计可能改善的是质量与训练稳定性。若推理仍使用422…和相同Noisy KV，模型调用数仍为24，DMD2本身不会直接提高FPS。GAN可能改善观感却损害配对PSNR或动作一致性，因此不能只按一项loss或FVD选择。

本次仅完成代码与原理核查，未启动新的DMD2训练。
