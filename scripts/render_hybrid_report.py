"""Write the shareable Markdown record from the exact imported results."""
from pathlib import Path
import json

repo = Path(__file__).resolve().parents[1]
data = json.loads((repo/'docs/results.json').read_text())
assert data['hybrid_evaluation']['episodes'] == 172
source = data['hf_source']
lines = ['# HF Hybrid Forcing checkpoint 比较','',
         f'新增 [Cosmos3-Edge hybrid step2400]({source["checkpoint_url"]})（非 EMA）。',
         '相同 172 条验证视频，固定 seed2026+index 与初始噪声、cached anchor、原始 RGB 参考和 TinyVAE BF16 流式解码。HF hybrid 与两版4222使用空 SFT 文本；旧学生保留发布的固定 action prompt。','',
         '| 模型 | FVD121 ↓ | FVD16 ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | A5000 FPS ↑ | L40S FPS ↑ |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for model in data['models']:
    def speed(gpu):
        value=data['hardware_benchmarks'][gpu]['models'][model['id']]['fps']
        return '排队中' if value is None else f'{value:.2f}'
    lines.append(f'| {model["name"]} | {model["fvd121"]:.2f} | {model["fvd16"]:.2f} | {model["lpips"]:.4f} | {model["psnr"]:.2f} | {model["ssim"]:.4f} | {speed("a5000")} | {speed("l40s")} |')
hybrid=next(m for m in data['models'] if m['id']=='hybrid')
lines += ['', 'HF hybrid 的 SSIM 均值为四模型最高，但 FVD 和 LPIPS 均差于原三模型。此处没有显示它在所有质量维度上的优势。','',
          'FPS 包含生成与 TinyVAE BF16 流式解码，排除加载、cached 首帧编码、CPU 转换、写文件和指标计算。每次请求内重建文本缓存；0000、0057、0114 各预热1次，再计时3次。hybrid 为新增一轮单卡测速，原三模型的已完成结果保留。','',
          '原生配置为 `[4,3,2,2,2,2,2,2,2,2]`、sink1/history2、absolute RoPE、历史记忆开启，共23次去噪＋11次 clean commit。完整加载344个LoRA、5个action、8个time、28个history张量；BF16推理，RoPE inv_freq保留FP32。', '',
          'HF训练标注为clips1k；student来自causal step12550，score来自bidir step31000。它与两版4222的训练数据、窗口、score初始化及步数不同；与本页验证集的训练重叠未核实，不能将差异单独归因于历史记忆。', '',
          f'固定官方 NavWorldGym runtime commit `{source["runtime_commit"]}`；HF revision `{source["revision"]}`。', '',
          '[三条常规场景四模型视频](../../../docs/assets/model-comparison.mp4) · [保留真实参考的视频](../../../docs/assets/model-comparison-with-reference.mp4)', '',
          '[雨天直行2.5m/s](../../../docs/assets/rain-comparison-2p5mps.mp4) · [慢行0.5m/s](../../../docs/assets/rain-comparison-0p5mps.mp4) · [静止](../../../docs/assets/rain-comparison-stop.mp4)', '',
          '上排：旧 stage3 student / 4222 clean；下排：4222+noisy kv reuse / 43222+hybrid forcing。雨天视频没有真实未来参考，不参与172条验证质量指标。', '',
          '质量分片作业：82730_0..3；A5000测速：82728；L40S测速：82729；CPU汇总：82734。分片只使用Slurm分配的GPU，并通过零已有计算进程检查。', '',
          '原生hybrid与旧学生使用120行真实动作；现有两版4222保持训练时121行含anchor占位的约定。三条新hybrid测速视频与质量评测视频的RGB hash一致。']
if hybrid.get('fvd_delta'):
    ci=hybrid['fvd_delta']
    lines += ['',f'相对旧学生的FVD121差值：{ci["observed_delta"]:.2f}；配对bootstrap 95%区间：[{ci["paired_percentile_95ci"][0]:.2f}, {ci["paired_percentile_95ci"][1]:.2f}]（512次）。']
(repo/'data/2026-10-25/hf-hybrid/report-zh.md').write_text('\n'.join(lines)+'\n')
print('Wrote hybrid report')
