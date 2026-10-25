# Hybrid Forcing 推理配置核查

当前核查没有发现窗口设错、漏载 history 权重或记忆未读取的问题。官方默认路径在片段0000重跑后，与原172条评测的未压缩RGB hash一致。

| 项目 | 官方配置 | 原评测 |
|---|---|---|
| 生成步数 | 4,3,2,2,2,2,2,2,2,2 | 一致；23次denoise＋11次clean commit |
| 历史窗口 | sink1 / history2 | 一致；保留anchor和最近2个历史块 |
| 位置编码 | absolute；记忆采用spatial RoPE | 一致；rotary inv_freq保持FP32 |
| 记忆读取 | 开启；fold vision_action | 一致；28层history权重完整加载 |
| 采样 | Euler；CFG1；shift5 | 一致；使用固定的共同初始噪声 |
| 文本条件 | 官方示例传入结构化prompt | 原比较使用空文本，条件不同 |

窗口中的history2是两个历史块，并非两帧。每个生成块为3个latent帧、对应12个新RGB帧。第4个生成块开始，block1离开窗口并折叠进记忆；最后一块直接看到anchor及blocks8、9，更早的blocks1..7由记忆承载。

在片段0000的第0、13、27层，63次具有已移出历史的读取全部产生非零输出贡献；关闭读取后贡献全部为0。改为完整历史后不发生移出，不会读取压缩记忆。前3个生成块在三种空文本设置下的latent完全相同。

## 固定十条视频的配对诊断

沿用预先确定的十个均匀索引：0、19、38、57、76、95、114、133、152、171。相同anchor、动作、初始噪声及TinyVAE BF16流式解码；31次新生成，另9条官方默认结果复用已完成评测。此处不是全量FVD重测。

| 设置 | LPIPS ↓ | PSNR ↑ dB | SSIM ↑ |
|---|---:|---:|---:|
| 官方窗口＋记忆，空文本 | 0.4693 | 15.49 | 0.4679 |
| 完整历史，空文本 | 0.4743 | 15.68 | 0.4702 |
| 官方窗口，关闭记忆，空文本 | 0.4723 | 15.52 | 0.4712 |
| 官方窗口＋记忆，通用结构化文本 | 0.4618 | 15.55 | 0.4687 |

结构化文本由官方minimal helper生成，内容仅为通用车载前视街景，不含具体场景或未来描述。它用于检查格式和非空条件的影响，并不能代表官方示例的完整场景prompt，也不能证明空文本就是FVD退步的原因。

172条的FVD121仍为269.73，不会用十条诊断的结果替换。HF checkpoint标注为clips1k、score step31000；两版4222使用4775条训练clip及score step30000，训练长度与窗口也不同。尚不能把质量差异单独归因于记忆模块或某个训练因素。

[官方 checkpoint 说明](https://huggingface.co/SichengMo-UCLA/NavWorldGym/blob/97c822da1be03ee19092c937aefd24b61190c960/edge-text-hybrid/step-2400/README.md) · [固定官方 runtime](https://github.com/DhlinV/NavWorldGym/tree/dec58fab57464baab30f664ee78769e5dc5fae54) · [完整核查数据](config-audit.json) · [原始诊断记录](config-audit-raw.json)

[公开研究日志](https://ceciliathebirb.github.io/research-log/)。原chatgpt.site管理接口返回404，当前Sites连接列出的项目为空；新网站从同一公开GitHub仓库main/docs发布。
