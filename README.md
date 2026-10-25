# cecilia-research-log

按日期整理的研究记录。每篇依次包含模型设计与训练细节、Qualitative + Quantitative results，以及 Findings。

[公开网页](https://cecilia-research-log.snowypearl2049.chatgpt.site)

## 2026-10-25 · Stage 3 蒸馏

比较三个现有 Cosmos3-Edge 模型：**旧 stage3 student**、**4222 clean**、**4222+noisy kv reuse**。

- `docs/`：可直接静态托管的网页、样式、脚本与观看视频。
- `docs/results.json`：精确质量指标与 L40S / A5000 实测吞吐。
- `data/2026-10-25/`：质量汇总、两种 GPU 各 27 次计时记录及来源说明。
- `scripts/`：生成报告的内容与脚本，以及对应 CoStream 评测脚本。

网站只展示优化后的推理结果。三个模型都使用直接单卡和请求内文本缓存；历史 clean / noisy KV 策略按各模型训练约定保留。FPS 包含生成和 TinyVAE BF16 流式解码，排除加载、首帧编码、CPU 转换、写 MP4 与质量指标。

质量表是 172 条 validation 的已完成评测（RTX A5000 / L40S 分片），并非分别在两种 GPU 上重跑的两套全量质量结果。双 GPU 速度表使用同 3 条固定视频、预热 1 次后各计时 3 次。

本轮速度作业为 L40S `82712`、A5000 `82713`，均通过空闲 GPU 检查。两次作业将 JSON 写进临时代码目录，成功退出后的清理移除了该文件；仓库中的速度记录从持久 Slurm 日志内的全部 27 条 JSON 计时及最终汇总恢复，并重新计算中位数与 FPS 逐一核验。`provenance` 保存恢复说明和原日志 SHA256。后续评测脚本已强制要求绝对输出路径，以免再次随临时目录清理。

## 更新报告

运行 `python3 scripts/build_report.py`，即可从仓库内的内容与数据重建 `docs/index.html`。先更新 `docs/results.json` 与 `data/`，再重建并提交。无需构建框架或外部依赖。

GitHub Pages 可将 `main` 分支的 `docs/` 作为静态站点来源。视频是压缩观看副本；指标使用未压缩生成 RGB。仓库不包含模型权重、训练视频数据集、服务器凭据或机器本地路径。
