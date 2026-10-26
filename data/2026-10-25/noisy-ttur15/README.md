# 4222+noisy kv reuse · TTUR1:5

正式训练运行中：2160/4800（45%），360次student＋1800次fake。截至 2026-10-25T23:16:19-07:00。

从原 causal Stage2 step12550 重新初始化，fake/teacher 使用 bidir step30000；不接续现有noisy1600。沿用SP4/DP1、有效batch1、seed0、两处目标匹配、AnyFlow forward + 2×DMD及学习率2e-6。本轮只消融更新比例，没有GAN。

目标为800次student＋4000次fake。原对照为800＋800。每300次总更新（50次student）验证和保存；为了适配A5000的24小时作业限制，拆为0→2400、2400→4800两段，通过完整模型及优化器状态续训，段间不重置学习率或更新计数。

正式训练作业：82756、82760。最终评测作业：82761（172条质量＋A5000优化测速）、82762（L40S优化测速）。

预检查82755已完成：训练4775条、验证172条，clip/session重叠均为0；真模型裁剪运行2次student＋10次fake，梯度隔离通过，portable adapter和DCP重载后生成逐值一致。裁剪检查不代表质量或全分辨率显存结果。

沿用相同seed及loader；更新比例变化会改变哪些样本及逐步随机数用于student，因此不是逐student更新完全相同样本的配对实验。最终按相同student更新数及GPU小时共同比较。

评测沿用原172条验证视频、固定初始噪声、空SFT文本及原始相机RGB参考。TinyVAE BF16流式，直接单卡＋请求内文本缓存；FPS不含加载、首帧编码和写文件。checkpoint2100已保存并由Box同步器确认上传；当前所有已记录训练loss和梯度均为有限值。验证loss为0.21309（step2100，辅助目标；不能替代FVD或生成质量）。尚无本轮新FVD或FPS结果。

[完整实验记录](experiment.json) · [DMD2原理与当前实现差异](../dmd2-noisy-kv-review.md)
