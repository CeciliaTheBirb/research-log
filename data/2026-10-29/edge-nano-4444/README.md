# Edge × Nano：完整172条结果

2026-10-30核验。质量作业84988已正常完成，耗时5:37:08，exit0；使用3×RTX A5000。
单卡L40S作业85009仍为PENDING(Resources)，未开始，没有L40S FPS。
没有设置定期检查或自动发布任务。

## 质量与速度

同一172条validation、170个源session；相同输入、动作、逐chunk初始噪声。
普通causal AnyFlow学生：Edge step2200、Nano step1070非EMA。每chunk四步、CFG1、shift5、完整clean历史、独立原生KV。
121帧、480×832、20FPS；TinyVAE step38000 BF16流式。FVD121含首帧，FVD16从未来120帧均匀采样16帧；配对RGB指标排除首帧。

| 路由 | FVD121↓ | FVD16↓ | LPIPS↓ | PSNR↑ dB | SSIM↑ | 3×A5000 FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEEE · 纯Edge | 231.96 | 288.44 | 0.43379 | 15.8076 | 0.50289 | 14.6982 |
| NNNN · 纯Nano | 180.42 | 239.32 | 0.38341 | 16.6175 | 0.53113 | 5.7700 |
| NEEE · Nano开头 | 229.70 | 292.96 | 0.41564 | 16.2383 | 0.51333 | 8.2750 |
| EEEN · Nano收尾 | 242.73 | 283.72 | 0.41568 | 16.0988 | 0.52013 | 8.2740 |
| EEEN · 最终历史诊断 | 242.73 | 283.72 | 0.41568 | 16.0988 | 0.52013 | 7.5887 |
| EEEN · Edge draft历史诊断 | 234.02 | 293.41 | 0.42096 | 15.9309 | 0.51323 | 7.5845 |

FPS是固定视频57、预热1次、计时3次的总耗时中位数对应吞吐，分子120个新帧。
包括文本缓存建立、rollout、模型切换/设备传输、活跃模型历史提交和TinyVAE解码；
不含加载、cached首帧编码、CPU RGB转换、写文件及质量指标。
Edge在CUDA0；Nano前18层和输入输出在CUDA1、后18层在CUDA2，两模型始终常驻。
纯模型只调用自己，其他模型闲置；这是3卡分层部署吞吐，不能称为单卡A5000或推算GPU成本节省。
两组诊断均额外计算Edge末步，不能用其FPS代替实际EEEN部署速度。

## Nano开头 vs 收尾

没有全面胜者：LPIPS几乎相同；FVD121点估计偏向开头、FVD16偏向收尾，两项配对区间都跨0。
PSNR偏向开头、SSIM偏向收尾。不能将某个均值排序推广为所有视频或全部质量指标的优势。

## 最终历史 vs Edge draft历史

最终历史改善配对画质，但没有明确FVD收益。两组展示均采用Nano末步输出；
唯一差别是Edge的历史提交自己的draft还是Nano最终latent；Nano总是提交最终结果。
每个模型通过自己的原生clean前向重建KV，没有跨模型复制KV，也没有梯度回传。
全视频LPIPS、PSNR、SSIM的区间均支持改善；后续chunk2–10与末段8–10平均LPIPS同样如此。
但FVD121点估计更高、FVD16更低，两个差值区间均跨0；不能声称整体视频质量稳定提升。

## 配对区间

源session成组配对bootstrap，RGB5000次、seed20261029；FVD512次、seed20261030。下表为前者减后者。

### Nano开头−Nano收尾

| 指标 | 差值 | 95%区间 | 前者胜出 |
| --- | ---: | --- | ---: |
| LPIPS↓ | -0.00004 | [-0.00488, +0.00448] | 76/172 |
| PSNR↑ dB | +0.1395 | [+0.0334, +0.2565] | 97/172 |
| SSIM↑ | -0.00680 | [-0.00963, -0.00408] | 56/172 |
| FVD121↓ | -13.03 | [-35.63, +16.55] | 集合指标 |
| FVD16↓ | +9.24 | [-16.69, +38.03] | 集合指标 |
| chunk2–10 LPIPS↓ | +0.00042 | [-0.00466, +0.00518] | 75/172 |
| chunk8–10 LPIPS↓ | +0.00468 | [-0.00078, +0.00993] | 74/172 |

### 最终历史−Edge draft历史

| 指标 | 差值 | 95%区间 | 前者胜出 |
| --- | ---: | --- | ---: |
| LPIPS↓ | -0.00528 | [-0.00727, -0.00325] | 111/172 |
| PSNR↑ dB | +0.1679 | [+0.1215, +0.2142] | 124/172 |
| SSIM↑ | +0.00690 | [+0.00527, +0.00861] | 136/172 |
| FVD121↓ | +8.71 | [-7.59, +25.51] | 集合指标 |
| FVD16↓ | -9.69 | [-31.55, +11.23] | 集合指标 |
| chunk2–10 LPIPS↓ | -0.00587 | [-0.00807, -0.00361] | 111/172 |
| chunk8–10 LPIPS↓ | -0.00584 | [-0.00953, -0.00207] | 104/172 |

这是反复使用的validation、每clip一个canonical seed的描述统计。
RGB/FVD bootstrap使用相同session配对抽样；逐chunk为pointwise区间，未作多重比较校正。
不保证外部学生从未见过这些数据，也不代替独立测试、多seed与长rollout验证。

## 一致性与复算

- 六组完整172条的index、CID、参考、seed、初始噪声hash和动作hash逐条匹配。
- 实际EEEN与最终历史shadow诊断的完整latent hash逐条一致。
- 两组历史诊断首chunk LPIPS一致；原生GPU预检确认首chunk逐值一致。
- 纯EEEE/NNNN与各自原生runner逐值一致；Nano18/18与20/16分层输出一致。
- 低秩样本空间FVD与原始协方差平方根FVD逐模型交叉核验，差值小于0.01；导出NPZ再次复算全部12个FVD值，通过。
- 模型payload hashes和已发布的预热重复测速记录与完整作业一致。

## 文件

- `validated-summary.json`：完整质量；`paired-analysis.json`：完整配对区间；`per-clip-quality.json`：逐clip指标。
- 原始I3D特征保留在服务器和本地输出目录，未上传公开站点；公开保留完整质量指标与配对统计。
- `analysis/analyze_results.py`：原始JSON/NPZ全量分析；`analysis/summarize.py`：独立FVD实现。
- `protocol.json`、`benchmark.json`、`parity.json`、`model-payload-sha256.json`：协议与一致性记录。
- `interim12.json`、`interim.json`、`paired-interim.json`：此前阶段快照归档，不作为完整结果。
- [模型路由调研](model-routing-research.md)：原始论文与尚未运行的下一步建议。

[公开完整实验页](https://ceciliathebirb.github.io/research-log/2026-10-29.html#quality-results)

[白天路口视频](../../../docs/assets/edge-nano-20261029/comparison-0000.mp4) · [夜间街道](../../../docs/assets/edge-nano-20261029/comparison-0057.mp4) · [树荫人行道](../../../docs/assets/edge-nano-20261029/comparison-0114.mp4)


## 雨后路口六视图 · 2026-10-30

新增 `rain-cases.json`：普通 Edge step2200 / Nano step1070，同一输入与 seed2026；2×RTX A6000，模型同时驻留，使用各自的完整 clean 历史。两段视频分别直行 2.5 m/s、0.5 m/s。上排 EEEE / NNNN / NEEN，下排 NEEE / EEEN / NNEE；六个画面均为生成结果。

NEEN 的原生 E/N 一致性检查覆盖两个生成 chunk，最大差为0；交付核对了模型、输入、动作、噪声、时间网格、解码、历史、GPU 部署和调用数。视频均为121帧、20 FPS播放、6.05秒，时间戳间隔0.05秒。权重与视频 SHA256 记录在 JSON，视频保存在本仓库 `docs/assets/edge-nano-20261029/`。NEEN 和 EEEN 使用 Nano 最终 latent 作为两个模型的历史内容；NNEE 则使用 Edge 最终 latent。没有 Edge draft 分支。

本次网页移除配对分析展示，既有归档未删除。新增的雨后视频没有并入172条参考质量指标，也没有产生新的预热 FPS。

当前两段六路视频已替换原五路加静态输入图版本。新增 NNEE 来自作业85502，原生 E/N 两生成 chunk 一致性检查最大差0；模型、输入和采样协议与其他路由一致。`rain-cases.json` 更新了视频哈希、NNEE 生成哈希和历史策略。
