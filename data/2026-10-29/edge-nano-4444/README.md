# Edge / Nano 四步混合推理：阶段记录

2026-10-29。作业84988运行中，节点bolei-gpu02，分配 **3×RTX A5000 24GB**。
Edge常驻CUDA0，Nano前18层及输入/输出位于CUDA1，后18层位于CUDA2。
没有量化、CPU offload、梯度计算或权重更新。单张L40S测速改为3小时短作业85009，
保留三条pilot、原生一致性检查、预热与三次重复测速，不重复172条全量质量评测。
原84980仅在仍排队时取消；85009最后确认状态为PENDING，尚无已验证结果。

10月29日20:15本轮再次读取时，本地SSH控制连接缺失，直连超时，未取得新进度。
本页质量数字保留此前验证过的12条快照，不能据此推断远端仍只完成12条。
用户明确要求**不设置定时检查或自动发布任务，等提问时再查看**。

使用普通因果AnyFlow学生：Edge step2200，Nano step1070非EMA。每个chunk四步，
shift5、CFG1、完整clean历史、两套独立文本/KV缓存、BF16流式TinyVAE step38000。
两模型共用实际时间网格、输入图、动作及逐chunk初始噪声。

## FPS：已完成预热与重复测速

固定验证视频57，预热1次、计时3次，报告总耗时中位数对应的FPS。
分子为120个新生成帧，包含去噪、模型切换/设备传输、文本缓存构造、历史提交与TinyVAE解码；
排除加载、cached anchor编码、CPU RGB转换、文件写入和质量指标。
两模型始终常驻；纯Edge或纯Nano只运行其自身去噪和历史提交，另一个模型闲置。
这是上述分卡配置的吞吐，不能称为单张A5000结果或直接推导GPU资源节省。

| 每chunk路由 | FPS |
| --- | ---: |
| EEEE：纯Edge | 14.6982 |
| NNNN：纯Nano | 5.7700 |
| NEEE：Nano开头 | 8.2750 |
| EEEN：Nano收尾、两边提交最终latent | 8.2740 |
| EEEN反馈诊断：额外计算Edge末步、提交最终latent | 7.5887 |
| EEEN反馈诊断：额外计算Edge末步、Edge提交自己的draft | 7.5845 |

诊断组的额外Edge末步用于固定计算路径，只改变Edge历史写入内容；其速度单独列出。

## 已完成的共同12条：非最终质量结果

索引0–9、57、114，完整121帧视频，配对质量排除已知首帧。
来自相同172条validation；外部学生历史训练数据与该集合的重叠尚未确认。
数字由`interim.json`快照汇总，后续全量结果可能改变排序。

| 路由 | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: |
| EEEE | 16.9296 | 0.5212 | 0.4202 |
| NNNN | 17.9998 | 0.5576 | 0.3643 |
| NEEE | 17.4734 | 0.5365 | 0.3960 |
| EEEN，最终历史 | 17.3223 | 0.5420 | 0.4014 |
| EEEN，Edge draft历史 | 17.1266 | 0.5343 | 0.4067 |

两种混合方案的平均LPIPS优于纯Edge，但仍不及纯Nano。Nano收尾的最终latent写回Edge历史后，
平均LPIPS由0.4067变为0.4014，12条中有8条改善。这只是当前样本趋势，尚不足以认定稳定收益。
两组反馈诊断第一chunk逐值相同，此后的差异来自Edge历史；不是把Nano的KV直接复制给Edge。

**FVD121 / FVD16仍待172条全部匹配完成后计算，不报告小样本FVD。**
正在运行的作业会自动生成全部质量指标和最终`summary.json`。

## 一致性检查

- 原生Nano的18/18与20/16层分配，两生成chunk逐值相同，最大差0。
- 新runner的EEEE、NNNN与各自原生生成逐值相同，最大差0。
- 加入Edge shadow末步不改变最终历史版本的latent；12条完整视频均hash一致。
- 两反馈组的首chunk一致，第二chunk能够因历史不同而变化。
- 已完成共同12条的初始噪声和动作hash逐条匹配。
- 空闲GPU检查通过；未终止或占用其他用户的GPU进程。

## 同步对比视频

上排：真实参考 / 纯Edge / 纯Nano。下排：Nano开头 / Nano收尾并提交最终历史 / Nano收尾但Edge保留draft历史。
每条6.05秒、20FPS，压缩文件用于观看，指标基于压缩前RGB。

- [白天路口 · 0](../../../docs/assets/edge-nano-20261029/comparison-0000.mp4)
- [夜间街道 · 57](../../../docs/assets/edge-nano-20261029/comparison-0057.mp4)
- [人行道 · 114](../../../docs/assets/edge-nano-20261029/comparison-0114.mp4)

原始记录：`settings.json`、`parity.json`、`benchmark.json`、`interim.json`。

## 模型切换调研与全量分析准备

- [原始论文、适配判断与下一步建议](model-routing-research.md)：TRACK、SDVG、T-Stitch、SRDiffusion、HybridStitch v3、连续扩散speculative sampling。
- `analysis/analyze_results.py`：仅在六组共同172条通过hash和参考匹配后计算完整FVD，按源session成组配对bootstrap；分别分析Nano开头/收尾和最终/draft历史，包括后续chunk。
- `../../../scripts/import_edge_nano.py`：一次性导入已验证的完整质量或L40S测速；检查FVD两种算法一致、模型payload、采样器、原生一致性和计时口径，不提交GPU作业、不循环、不推送Git。
- L40S发布前还需读取实际输出，并通过上述检查；没有用其他GPU的速度推算。

[公开实验页](https://ceciliathebirb.github.io/research-log/2026-10-29.html)
