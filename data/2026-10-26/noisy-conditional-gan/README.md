# 1:1 Noisy KV + 动作条件视频GAN

保留修正版Noisy KV 1:1基线，单独增加GAN监督。正式训练作业 **83083** 已在4×RTX A5000启动。计划800次student＋800次fake，结束后依次执行A5000质量/速度作业83084和L40S速度作业83085。尚无新质量或FPS结论。

从同一causal Stage2 step12550重新开始；fake score和冻结teacher仍从bidir step30000加载。维持4×A5000、SP4/DP1、batch1、seed0、422…网格、Noisy KV、历史噪声匹配、shifted-uniform score时间、原AnyFlow和DMD损失、student/fake学习率2e-6。

预算为800次student＋800次fake score；新增判别器随fake阶段更新800次。fake score与GAN判别器是两个独立网络。student目标是 `L_forward + 2 L_DMD + λ L_GAN`；λ在前50次student更新从0.00002升到0.001。

判别器是随机初始化的轻量时序CNN，读取完整的归一化Cosmos latent未来序列、首帧latent、相邻帧差及时间位置；每个latent时间的projection分支读取对应四次RGB动作转移。三个时空卷积层和一个膨胀时间卷积覆盖约31个latent时间位置。它是在latent空间做的视频GAN，不是DMD2官方的预训练score特征判别器。

使用非饱和logistic GAN；判别器AdamW lr=1e-4、β=(0,0.99)，每16次D更新做gamma1的lazy R1。真实与生成latent采用相同精度并加std0.01的同方差instance noise，避免BF16真实/FP32生成精度捷径。真实样本仅来自4775条训练clip，不读取172条validation作训练监督。

student阶段通过冻结判别器的输入梯度更新student；fake阶段只更新fake score及独立判别器。判别器副本梯度按DP/SP规则同步，checkpoint保存其权重与优化器。GAN初始化与噪声使用独立随机数流。

训练无需VAE解码；评测仍用TinyVAE BF16流式、直接单卡与请求内文本缓存。判别器不进入推理，调用数仍为24，不能预设FPS或画质会改善。

评测沿用相同172条validation、canonical初始噪声、空SFT文本和原始RGB参考；报告FVD121/FVD16、LPIPS、PSNR、SSIM与配对差值区间。A5000/L40S各自重复测速，排除加载、首帧编码、CPU转换、写文件和指标。基线输出保留，测速另在同一作业重跑1:1对照。

实现与配置见[code](code/)；最新作业及状态见[experiment.json](experiment.json)。

预检记录：配置及数据分离检查、22项单元/回归测试、单卡4次更新、独立新进程精确恢复、四卡全分辨率2次更新均通过。四卡使用 `[1,48,31,30,52]` latent，rank0峰值allocated显存15.52GiB，强制每次D更新执行R1，确认其反传有效。见[四卡验证](four-gpu-full.json)、[单卡训练](single-gpu-train.json)、[精确恢复](single-gpu-reload.json)。这些是代码正确性检查，不是质量评测。

首个单卡预检83052因诊断打印重复loss字段退出，已修复；首次四卡83057在模型构建时rank3 SIGSEGV，未开始训练，根因尚未确认。之后使用相同核心训练代码，在另一正常调度节点启用崩溃堆栈、降低OMP线程并增加主机内存预留，83076完整通过。正式训练83083使用这组通过的运行设置；首次GPU guard确认四张分配卡无既有计算进程。

2026-10-26 21:00（洛杉矶）状态：**718/1600步，359次student＋359次fake/GAN更新**，约45%；已运行6小时21分。step700的adapter、训练state及`video_gan.pt`已通过Box目录实际核对。所有已记录数值有限，rank0峰值allocated显存17.44GiB；R1已执行22次。辅助验证loss从step100的0.21846降至step700的0.21307，这是固定4个validation batch的forward损失，不是172条视频FVD或GAN质量结论。warmup完成后的稀疏记录中，加权GAN/DMD输出latent梯度RMS比为1.17%–2.70%；说明GAN有实际梯度贡献，尚不能据此判断画质收益。按31.57秒/总更新估计还需约7.7小时训练，随后才开始完整评测，排队和评测另计。原始逐步指标见[training-metrics.jsonl](training-metrics.jsonl)，摘要见[progress.json](progress.json)。
