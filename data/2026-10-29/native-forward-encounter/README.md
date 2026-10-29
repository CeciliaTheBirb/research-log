# 原生 Cosmos3 Super / Nano / Edge：前进＋迎面行人

2026-10-29归档；实验ID `native-forward-encounter-20261028`。本轮只推理，没有训练。

36条输出全部完成：每模型2张首图 × 2 seeds × 3事件。事件为无新增前景行人、迎面从机器人左侧经过、迎面从机器人右侧经过。所有条件均请求机器人以0.5 m/s直行6秒；120行AV9动作，每行 `[0,0,0.025,1,0,0,0,1,0]`。前进3米是输入要求，未从视频测出实际位移。

## 结果

完整视频位于 `docs/assets/native-forward-encounter/`：四个18.15秒合辑，另有12个按条件切分的6.05秒并排视频。左Edge／中Nano／右Super。网页为 `docs/2026-10-29.html`。保留两个seed全部结果，未筛选成功样例。

人工审看0、1.5、3、4.5、6秒的帧及其相邻时刻：雨后路口三者背景近似静止，行人经常横穿或走远；Super夜景seed2026有明显前进，但seed2027没有稳定复现，且近景物体有形变。三者均未稳定完成迎面靠近并从指定一侧经过。此为定性观察，未报告自动成功率。

这不能单独说明Super更好或更差，也不能把原因直接归到未微调的机器人域。本轮使用官方基础权重、CoStream移植后端；完整checkpoint的BF16官方运行时对齐尚未完成。固定文本改变动作的独立动作对照也尚未完整完成；本轮文本与动作同时请求前进。

## 条件与核查

- 原生单时间、双向50步UniPC，FD guidance=1、shift=5、domain=av，action JSON文本；无adapter、无LoRA、无causal历史、无Noisy KV reuse。
- 480×832，121帧，20 FPS；TinyVAE step38000 BF16流式。
- 重复首图到121帧再编码，仅latent第0帧作条件；无未来真实帧输入。
- 跨模型图像、anchor、CPU FP32初始噪声、原始动作、事件文本与manifest一致；无文本截断。
- 三套官方权重均做文件完整性和完整tensor键名／形状检查。
- 8组缩小模型FP32与NavWorldGym native vendor的FD/I2V数值对齐，最大误差0；这不等于完整BF16权重对齐。
- 120行数值与项目原有AV9原语约定一致；尚不能仅据格式正确认定官方完整运行时行为一致。
- Slurm 84721 / 84722 / 84723全部退出0；标准分配GPU guard检查通过。各模型输出已由作业同步到Box实验目录。

`summary.json` 汇总模型固定revision、硬件和实际耗时；`results.json` 是36条原始输出记录；`*-settings.json` 为去除服务器绝对路径的运行配置；`audit.json` 验证输入一致性及视频hash；`manifest.json` 是输入设计；`code/`保存此次评测与核查脚本，需CoStream环境运行。

计时包含生成＋TinyVAE解码，排除加载、首帧编码、CPU转换和写文件。每个条件只跑一次，12条不同条件取中位数，没有重复测速。Edge 1×A5000且有文本缓存，Nano 2×A5000、Super 4×A6000且无文本缓存；不能视为同硬件速度排名，也不能同4222学生FPS直接比较。

两场景两seed是小规模控制诊断，不估计总体FVD。FVD不要求逐视频配对，但需要合适的真实／生成样本集合。反事实条件没有配对真实未来，因此无PSNR／SSIM；旧172条validation指标不属于本轮三种原生模型。

## 下一步的判别性检查

先用官方运行时的已知有效动作与同一机器人图像对照，确认完整权重推理与动作坐标；再固定文本比较静止／前进动作，并独立比较无动作I2V与FD的行人控制。当前证据不足以立即决定第一阶段训练。

官方参考：[Super固定版本模型卡](https://huggingface.co/nvidia/Cosmos3-Super/blob/f543c56225b2e04d0ad141e29655be3a45d9c455/README.md)，[Diffusers固定版本输入实现](https://github.com/huggingface/diffusers/blob/40e551a4380c0a48333f3eeb032ed7da1f361297/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py)。官方参考说明接口；本轮实测结果来自本目录记录。
