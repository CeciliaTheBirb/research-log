# 已训练 Edge / Nano 的推理时模型切换

调研日期：2026-10-29。以下是原始论文与适配建议，不是新实验结果。
当前实验仍是普通 causal AnyFlow 四步学生，TinyVAE BF16 流式、独立原生 KV、固定动作与初始噪声。

| 原始来源 | 方法 | 适配判断 |
| --- | --- | --- |
| [TRACK, 2026-09](https://arxiv.org/html/2609.30096v1) | 沿大模型参考轨迹，在同一 latent/time/condition 比较预测；离线选择分歧小的步骤给小模型，部署每步只执行一个模型，可非连续切换。 | 最适合近期开销小的扩展。已覆盖 Cosmos3 Super/Nano、Super/Edge 和少步视频生成；未验证当前动作条件 causal student。建议补 ENEE/EENE，并按 chunk、动作校准。 |
| [SDVG, 2026-04](https://arxiv.org/html/2604.17397v1) | 小模型起草整块，ImageReward 最差帧分数决定接受或大模型重生成；首块强制大模型。drafter 始终保留自己输出的历史。 | 优先测试按 chunk 分配 Nano。原论文的画质路由不保证机器人动作或几何正确；拒绝 draft 需恢复流式 decoder 状态。与本实验“Edge 采用最终 Nano 历史”是不同策略，不能预设孰优。 |
| [T-Stitch, 2024](https://t-stitch.github.io/) / [SRDiffusion, 2025](https://arxiv.org/abs/2505.19151) | 分别先小后大、先大后小；SRDiffusion 由高噪声大模型 sketch 转到小模型 render。 | EEEN/NEEE 都有相关先例。不能从名称推断当前四步学生哪个顺序更好。SRDiffusion [官方代码](https://github.com/alibaba/SRDiffusion)有自适应切换阈值。 |
| [HybridStitch v3, 2026-10-14](https://arxiv.org/html/2603.07815v3) | token 级选择采用哪个模型更新；共享 latent/sequence，每个模型自己的逐层 KV buffers；仅执行选中查询。 | 行人/障碍物 ROI 的 Nano 修正值得探索，但要改部分 query/FFN 执行与缓存维护。两网全算再混合没有这种节省。论文是图像和 diffusion LLM，尚非当前流式机器人视频。 |
| [Accelerated Diffusion Models via Speculative Sampling, 2025](https://arxiv.org/html/2501.05370v2) | 随机转移核下，用反射最大耦合做 draft/verify，可保持目标分布。 | 论文讨论明确排除直接用于确定性 sampler；当前四步 Euler 不可直接套用“无损 speculative decoding”。 |

## 当前系统建议，尚未运行

1. 一个 Nano 调用预算：NEEE、ENEE、EENE、EEEN。先在独立校准子集选择，再在保留集确认；不能在已反复比较的 172 条上调好后称其为独立测试。
2. 跨 chunk 同 Nano 调用预算：四个 NEEE chunk 对比一个 NNNN 加三个 EEEE。需要计入额外 prefill/commit 和模型常驻内存；NFE 相同不代表 FPS 相同。
3. 关键 chunk 路由：用已知动作与已生成历史决定是否调用 Nano，并与同预算的随机/周期策略对比。不能用真实未来作在线路由输入。
4. 后续再考虑 token/区域路由；两模型原生 KV 不可互相直接复制。

当前每块时间网格为 `[1, 0.9375, 0.8333333, 0.625, 0]`，四个区间宽度不等。
“Nano 最后一步只修细节”并非这个 sampler 的已证事实。

模型切换本身已经有先例。潜在研究问题应落在动作条件、长时因果历史、
修正何时写回、路由开销与固定延迟预算的共同优化；需要实验支持，不能预先声称新颖性或质量收益。

## 172 条结果如何回答问题

- 逐条匹配动作、初始噪声、CID、参考视频；shadow 分支需与部署版本 latent hash 相同。
- 开头 vs 收尾：FVD121、FVD16、LPIPS、PSNR、SSIM；给出配对差值。
- 最终 vs draft 历史：固定额外 Edge 末步诊断路径，第一 chunk 相同；看后续第 2–10、末段第 8–10 chunk。
- 按源 session 成组做配对 bootstrap，报告区间和胜出条数。172 clips 来自 170 sessions。
- 这是反复使用的 validation 上的描述统计，不能替代独立测试或多 seed 稳定性验证。
- 单 L40S 与三 A5000 分开报告，诊断 FPS 不能冒充实际部署 FPS。

仅一次性分析和发布。按用户要求，不设置自动检查、定时任务或自动推送任务。
