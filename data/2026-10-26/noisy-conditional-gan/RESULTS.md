# 1:1 Noisy KV + 动作条件视频 GAN：完成结果

2026-10-27，洛杉矶。训练83083、172条质量及A5000重复测速83084均已完成。4×RTX A5000训练14小时06分28秒，约56.43 GPU小时；800次student、800次fake score、800次独立GAN判别器更新。L40S测速83085已完成。

在相同172条验证视频上，小权重GAN改善配对RGB指标，FVD点估计略低，但FVD差值区间跨0。不能据此宣称完整DMD2有效或已经训练收敛。

| 指标 | Noisy KV / 1:1基线 | Noisy KV / 1:1 + GAN |
|---|---:|---:|
| FVD121 ↓ | 215.3470 | 212.6418 |
| FVD16 ↓ | 278.8466 | 277.7094 |
| LPIPS ↓ | 0.41504 | 0.40681 |
| PSNR ↑ dB | 16.1497 | 16.3094 |
| SSIM ↑ | 0.51416 | 0.51951 |
| 同作业A5000 FPS ↑ | 25.1165 | 25.1106 |
| 同作业L40S FPS ↑ | 57.5283 | 57.6454 |

差值定义为GAN减基线；95%配对bootstrap区间：

| 指标 | 均值差值 | 95%区间 |
|---|---:|---:|
| FVD121 | −2.7052 | [−14.6248, +7.2359] |
| LPIPS | −0.008233 | [−0.010288, −0.006252] |
| PSNR dB | +0.159741 | [+0.114003, +0.205375] |
| SSIM | +0.005358 | [+0.004067, +0.006633] |
| temporal difference L1 | −0.000239 | [−0.000399, −0.000079] |

FVD121区间使用512次配对重采样。172条validation在实验间重复使用，并非独立的新测试集；还需要独立保留测试和更长rollout验证。该结果仅适用于这版latent CNN GAN及权重0.001，不能等同于官方DMD2实现。

## 评测核对

两版Noisy均为24次模型调用、相同历史策略与动作输入约定、相同空SFT文本。逐条核对clip、seed、初始噪声hash、文本token数及hash；从172条原始记录重新计算RGB均值。GAN与旧学生使用的参考特征完全一致；基线既有跨GPU参考特征差异通过既有容差检查，所有FVD统一使用canonical参考。详情见[evaluation-audit.json](evaluation-audit.json)。

完整统计见[summary.json](summary.json)、[FVD区间](fvd-uncertainty.json)、[172条逐视频指标](episode-metrics.jsonl)。

## 速度与视频

同一作业83084、同一张A5000：旧stage3 student 14.7318 FPS；1:1基线25.1165 FPS；+GAN 25.1106 FPS。三者均为直接单卡＋请求内文本缓存＋TinyVAE BF16流式。每条0000/0057/0114各预热1次、计时3次；27条原始计时均已重算中位数。排除加载、首帧编码、CPU转换、MP4写入及质量指标；包括文本缓存建立、rollout与解码。见[a5000-benchmark.json](a5000-benchmark.json)。

报告总表保留1:1基线历史轮次26.19 FPS，评估本轮速度变化应使用以上同作业数据。GAN判别器不进入推理，观测到的速度差异约0.02%，无需据此推断速度退化。

[三个场景的四面板同步视频](../../../docs/assets/noisy-gan-comparison.mp4)：上排真实参考、旧学生；下排Noisy KV 1:1、Noisy KV 1:1 + GAN。121帧×3，20 FPS，18.15秒；MP4仅供观看，质量指标使用未压缩RGB。[视频输入hash与拼接记录](video-provenance.json)。

同一L40S作业83085：旧学生35.2326 FPS，1:1基线57.5283 FPS，+GAN 57.6454 FPS。同样27条原始计时均已重算，见[l40s-benchmark.json](l40s-benchmark.json)。

## 训练与 checkpoint

所有记录值有限，rank0峰值allocated显存17.45 GiB；50次R1。固定4个validation batch的辅助forward loss由0.21846变为0.21178，最低step1200为0.21105，后段仍有波动。稀疏记录的warmup后加权GAN/DMD输出latent梯度RMS比为1.17%–12.02%；这不是完整参数梯度比例。详见[progress.json](progress.json)、[完整训练指标](training-metrics.jsonl)。

[step1600 checkpoint](https://app.box.com/folder/431510569802)、[student所在adapter文件](https://app.box.com/file/2552548393910)、[metadata](https://app.box.com/file/2552545801742)已实际列目录核对，state内有`video_gan.pt`。推理使用匹配Cosmos3-Edge底座与student adapter，GAN判别器不参与推理；Box保留原有访问权限。
