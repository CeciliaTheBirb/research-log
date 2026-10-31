# cecilia-research-log

按日期整理的研究记录。每篇依次包含模型设计与训练细节、Qualitative + Quantitative results，以及 Findings。

[公开网页](https://ceciliathebirb.github.io/research-log/)

网站由本仓库 `main` 分支的 `docs/` 自动发布到 GitHub Pages。原 `chatgpt.site` 地址目前仍能浏览旧版，但当前 Sites 管理连接看不到该项目，读取原项目 ID 返回 404；新的公开地址与本仓库同步。

发布约定（2026-10-25）：后续研究日志与进度网页统一通过本仓库的 GitHub Pages 发布，固定公开地址为 https://ceciliathebirb.github.io/research-log/ 。不再使用 Sites / `chatgpt.site` 作为发布渠道。

## 2026-10-25 · Stage 3 蒸馏

比较四个 Cosmos3-Edge 模型：**旧 stage3 student**、**4222 clean**、**4222+noisy kv reuse**、**43222+hybrid forcing**（HF Edge step2400）。

- `docs/`：可直接静态托管的网页、样式、脚本与观看视频。
- `docs/results.json`：精确质量指标与 L40S / A5000 实测吞吐。
- `data/2026-10-25/`：质量汇总、两种 GPU 各 27 次计时记录及来源说明。
- `scripts/`：生成报告的内容与脚本，以及对应 CoStream 评测脚本。

网站只展示优化后的推理结果。四个模型都使用直接单卡和请求内文本缓存；历史 KV 与记忆策略按各模型原生配置保留。FPS 包含生成和 TinyVAE BF16 流式解码，排除加载、首帧编码、CPU 转换、写 MP4 与质量指标。

质量表是 172 条 validation 的已完成评测（旧学生与 HF hybrid 使用 RTX A5000，两版4222使用 A5000 / L40S 分片），并非分别在两种 GPU 上重跑的两套全量质量结果。双 GPU 速度表使用同 3 条固定视频、预热 1 次后各计时 3 次；HF hybrid 的 L40S 测速已完成，为48.16 FPS。

本轮速度作业为 L40S `82712`、A5000 `82713`，均通过空闲 GPU 检查。两次作业将 JSON 写进临时代码目录，成功退出后的清理移除了该文件；仓库中的速度记录从持久 Slurm 日志内的全部 27 条 JSON 计时及最终汇总恢复，并重新计算中位数与 FPS 逐一核验。`provenance` 保存恢复说明和原日志 SHA256。后续评测脚本已强制要求绝对输出路径，以免再次随临时目录清理。

## 更新报告

运行 `python3 scripts/build_report.py`，即可从仓库内的内容与数据重建 `docs/index.html`（10.25）、`docs/2026-10-27.html`（10.27）和 `docs/2026-10-29.html`（10.29）。各页左侧栏均可按日期切换。先更新 `docs/results.json` 与 `data/`，再重建并提交。无需构建框架或外部依赖。

GitHub Pages 可将 `main` 分支的 `docs/` 作为静态站点来源。视频是压缩观看副本；指标使用未压缩生成 RGB。仓库不包含模型权重、训练视频数据集、服务器凭据或机器本地路径。

## 雨后路口展示与 checkpoint

页面采用 Notion 风格的侧栏与文档排版。雨后路口四模型对比可切换直行 2.5 m/s、慢行 0.5 m/s 和静止。原三模型使用已完成的诊断输出，HF hybrid 新增对应动作视频；不从这些展示视频取吞吐数字。动作与出处见 `data/2026-10-25/rain/` 和 `data/2026-10-25/hf-hybrid/`。这组单图展示不参与 172 条验证集的质量统计。

`docs/checkpoints.json` 和页面最后列出 clean step3200、修正版 noisy step1600 的 Box checkpoint 与 adapter 文件链接，已通过 Box 目录及实际 meta.json 核对；保留 Box 原有权限。

## 新增 HF Hybrid Forcing 比较

新增 `SichengMo-UCLA/NavWorldGym/edge-text-hybrid/step-2400`（Cosmos3-Edge，非 EMA），固定 HF revision `97c822da1be03ee19092c937aefd24b61190c960` 和官方 NavWorldGym runtime `dec58fab57464baab30f664ee78769e5dc5fae54`。完整加载 LoRA、action、time、history 四组权重；原生 4,3,2,2,…，sink1/history2，历史记忆开启。

沿用相同 172 条验证视频、固定初始噪声、cached anchor、原始 RGB 参考和空 SFT 文本。新模型原生接口接收 120 行真实动作，TinyVAE BF16 流式解码，单卡＋请求内文本缓存。三条固定视频的测速与质量运行 RGB hash 一致。该模型训练数据标注为 clips1k，与当前验证视频的重叠未核实，不能称作对该外部 checkpoint 已确认未见过的测试集。

`data/2026-10-25/hf-hybrid/` 保存来源、配置、完整质量汇总、每卡重复测速与视频来源。新 hybrid 在 A5000 上的质量分片作业为 82730_0..3，测速作业为 82728；L40S 测速作业为 82729（未完成时表中标明排队，不填估算值）。原三模型的已完成质量与速度结果保留，hybrid 为新增一轮单卡测速。

四模型视频上排为旧 stage3 student / 4222 clean，下排为 4222+noisy kv reuse / 43222+hybrid forcing；雨天可切换三种动作。原三模型与真实参考视频保留在 `docs/assets/model-comparison-with-reference.mp4`。运行 `python3 scripts/import_hybrid.py` 后再运行 `python3 scripts/build_report.py` 更新表格。评测脚本供在 CoStream 环境内运行，不包含模型底座或数据。

[Hybrid 推理配置核查](data/2026-10-25/hf-hybrid/config-audit-zh.md) 记录官方窗口/记忆参数对照，以及固定十条视频的完整历史、关闭记忆和通用结构化文本诊断。原172条无caption成绩与这份诊断分别保留。运行 `python3 scripts/import_hybrid_audit.py` 校验原始诊断记录并更新摘要，再重建网页。

[Noisy KV 1:5 TTUR 实验](data/2026-10-25/noisy-ttur15/README.md) 已完成800次student＋4000次fake、172条质量评测及A5000 / L40S优化后测速；L40S为57.57 FPS。FVD121为221.91（1:1为215.35，差值95%区间跨0），本轮未显示整体收益。同作业A5000为25.76→25.73 FPS。三个场景的同步对比视频、指标、训练成本和Box checkpoint4800已归档并发布到[研究日志](https://ceciliathebirb.github.io/research-log/#ttur-results)。这只测试更新比例，没有GAN，不代表完整DMD2。

## 2026-10-27 · Noisy KV + 动作条件视频 GAN

[独立10.27页面](https://ceciliathebirb.github.io/research-log/2026-10-27.html)记录保留1:1基线的小权重GAN实验；左侧日期栏可返回10.25。正式训练83083已完成，4×A5000、800次student＋800次fake/GAN更新，耗时14小时06分28秒。λ=0.001，前50次student渐增，判别器不进入推理。

172条质量评测已完成：LPIPS 0.4150→0.4068、PSNR 16.15→16.31 dB、SSIM 0.5142→0.5195，配对差值区间均不跨0；FVD121 215.35→212.64，但差值95%区间[−14.62,+7.24]跨0。同作业A5000 25.12→25.11 FPS。完整统计、三个场景的同步视频、训练配方与最终Box checkpoint见[实验记录](data/2026-10-26/noisy-conditional-gan/RESULTS.md)。

## 2026-10-29 · 已训练 Edge × Nano 四步混合推理

[10.29 实验页](https://ceciliathebirb.github.io/research-log/2026-10-29.html)为全新的普通学生混合推理实验；此前撤下的原生base模型测试没有恢复。
使用Edge step2200、Nano-v0.1 step1070，比较EEEE、NNNN、NEEE、EEEN，以及计算路径匹配的Edge历史反馈对照，不更新权重。

作业84988已在3×RTX A5000运行：Edge占1卡、Nano分层占2卡；GPU空闲检查、原生逐值一致性和历史反馈隔离均通过。
固定视频57预热1次、计时3次，EEEE为14.70 FPS，NNNN为5.77，NEEE与EEEN均约8.27。包括模型切换、历史提交及TinyVAE BF16流式解码，排除加载、cached首帧编码、CPU转换、写文件与质量指标。
这些是三卡部署的吞吐，不是单张A5000结果；单卡L40S作业84980仍排队。

本次网页发布共同完成12/172条的阶段质量快照与三组同步视频，完整FVD尚待全部172条完成。页面不是服务器日志的实时镜像。
数据及协议在[data/2026-10-29/edge-nano-4444](data/2026-10-29/edge-nano-4444/)，视频均在`docs/assets/edge-nano-20261029/`，生成逻辑为`scripts/report_edge_nano.py`。


### 10.29 页面更新 · 2026-10-30

页面撤下配对分析栏目、区间表、逐 chunk 图及其下载入口；既有原始统计归档保留。新增雨后路口 2.5 m/s 与 0.5 m/s 六视图：上排 EEEE / NNNN / NEEN，下排 NEEE / EEEN / NNEE；六个画面均为生成结果。视频与封面位于 `docs/assets/edge-nano-20261029/rain-six-view-*`，来源、权重哈希、历史策略和交付检查见 `data/2026-10-29/edge-nano-4444/rain-cases.json`。

雨后案例使用 2×RTX A6000，属于独立视觉诊断；没有把其播放帧率当作模型吞吐，也未将其加入原 172 条 validation 指标。Findings 记录 Nano 开头的运动表现、收尾的局部画面改善，以及当前整体仍不及四步全 Nano 的观察。

两段雨后视频已替换为包含 NNEE 的最新六路版本，原五路加静态输入图版本已从页面和当前视频文件中替换。NNEE 来自作业85502，历史提交 Edge 最终 latent；视频与封面使用版本参数避免浏览器继续读取旧缓存。
