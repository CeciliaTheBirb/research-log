# 4222+noisy kv reuse · TTUR1:5

训练与172条质量评测已完成；A5000测速已完成，L40S测速排队。截至 2026-10-26T10:42:51-07:00。

已完成800次student＋4000次fake，共4800次总更新。4×RTX A5000、SP4/DP1，有效batch1。两段作业82756、82760均成功；第二段完整恢复第一段step2400的模型与优化器。训练占卡约13小时24分，约53.6 GPU小时（包含加载、保存、同步）。checkpoint4800已上传Box。

与修正版1:1 noisy对照相同：从causal Stage2 step12550重新初始化，fake/teacher为bidir step30000，seed0，4222网格，Noisy KV及历史噪声匹配，shifted uniform score时间，AnyFlow forward + 2×DMD，学习率2e-6。本轮只改更新频率及对应的保存/验证频率，没有GAN。

| 指标 | 1:1 noisy | 1:5 TTUR |
|---|---:|---:|
| FVD121↓ | 215.34702 | 221.90746 |
| FVD16↓ | 278.84656 | 275.86709 |
| LPIPS↓ | 0.41504 | 0.42588 |
| PSNR↑ | 16.14966 | 15.97402 |
| SSIM↑ | 0.51416 | 0.50945 |

FVD121差值为+6.56，配对bootstrap95%区间[-7.98,+15.77]跨0。因此未显示整体收益，不能称为已确认显著退步；FVD16略有改善。当前不据此替换1:1基线，也不能据此否定完整DMD2。

同作业A5000优化后测速：1:1=25.76FPS，1:5=25.73FPS。L40S作业82762仍排队，不填估算值。

全部172条对照已检查CID、seed、初始噪声hash、文本token hash和24次模型调用。训练4775条与验证172条的clip/session重叠均为0。训练日志逐步核对1:5比例、连续1至4800的更新数和有限loss/梯度。

沿用同一loader和seed；改更新比例后，进入student的样本和逐步随机数也会改变，并非逐student更新使用完全相同样本的配对训练。800次student预算相同。

质量基于未压缩生成RGB与原始相机RGB，172条验证、canonical noise、空SFT文本。速度为直接单卡＋请求内文本缓存＋TinyVAE BF16流式，不含加载、首帧编码、CPU转换、写文件和指标。

[Checkpoint4800](https://app.box.com/folder/431189754989) · [推理adapter](https://app.box.com/file/2550530063317) · [质量汇总](summary.json) · [测速原始记录](a5000-benchmark.json) · [核查记录](evaluation-audit.json) · [视频来源](video-provenance.json)

2026-10-27更新：L40S作业82762已完成。同作业Noisy 1:1基线57.76 FPS，TTUR1:5为57.57 FPS；原始计时见[l40s-benchmark.json](l40s-benchmark.json)，精确值以JSON为准。
