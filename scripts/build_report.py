"""Build the date-indexed research log using only repository-local assets/data."""
from pathlib import Path
import html
import json

r = Path(__file__).resolve().parents[1]
s = json.loads((r / 'docs/results.json').read_text())
training = (r / 'scripts/training-content.html').read_text()
quality = (r / 'scripts/quality-table.html').read_text()
protocol = (r / 'scripts/protocol-content.html').read_text()
fps_rows = []
for m in s['models']:
    cells = []
    for gpu in ['l40s', 'a5000']:
        b = s['hardware_benchmarks'][gpu]
        v = b['models'][m['id']]['fps']
        cells.append(f'<td>{v:.2f}<small>FPS</small></td>' if v is not None else '<td class="pending">正在补测<small>Slurm 82713</small></td>')
    fps_rows.append('<tr><th scope="row">' + html.escape(m['name']) + '</th>' + ''.join(cells) + '</tr>')
page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cecilia-research-log · 2026.10.25</title>
<meta name="description" content="Cecilia 的研究日志：模型设计与训练细节、定性与定量结果、Findings。Stage 3 蒸馏，L40S / A5000 优化后推理结果。">
<meta name="theme-color" content="#faf9f6"><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="style.css"><script defer src="app.js"></script></head>
<body class="research-log"><a class="skip" href="#log-2026-10-25">跳到日志正文</a>
<header class="log-masthead"><a href="#" class="log-brand">cecilia<span>—</span>research-log</a><div class="header-links"><a href="https://github.com/CeciliaTheBirb/research-log" target="_blank" rel="noopener">GitHub ↗</a><span>RESEARCH NOTES / 2026</span></div></header>
<div class="log-layout"><aside class="date-sidebar"><div class="sidebar-label">Date</div><nav aria-label="按日期浏览"><a class="date-link active" href="#log-2026-10-25" aria-current="page"><span>10.25</span><small>2026 · Stage 3 蒸馏</small></a></nav><div class="entry-nav"><a href="#model-design">01　模型设计</a><a href="#results">02　Qual + Quant</a><a href="#findings">03　Findings</a></div><div class="sidebar-bottom">WORLD MODEL<br>EXPERIMENT JOURNAL</div></aside>
<main class="log-content" id="log-2026-10-25"><header class="entry-header"><div class="eyebrow">OCTOBER 25, 2026 / EXPERIMENT LOG 001</div><h1>Stage 3 蒸馏</h1><p class="entry-deck">从 Stage 2 初始化，训练 4222 clean 与 4222+noisy kv reuse。记录模型设计、质量与速度结果，以及当前发现。</p><div class="entry-tags"><span>Cosmos3-Edge</span><span>4775 train / 172 validation</span><span>TinyVAE BF16 流式</span></div></header>
<section id="model-design" class="log-section">@@TRAINING@@</section>
<section id="results" class="log-section"><div class="section-head"><div><div class="eyebrow">02 / QUALITATIVE + QUANTITATIVE RESULTS</div><h2>定性与定量结果</h2></div></div>
<p class="section-intro">只展示优化后的部署结果。旧 stage3 student 已使用原生直接单卡＋文本缓存；两版新模型也开启直接单卡＋文本缓存。模型历史 KV 的 clean / noisy 策略按各自训练方案保留。</p>
<h3 class="result-title">Qualitative / 同步对比视频</h3><div class="viewer-head"><p>上排：真实视频、旧 stage3 student<br>下排：4222 clean、4222+noisy kv reuse</p><a href="assets/model-comparison.mp4" download>下载 MP4 ↓</a></div>
<div class="video-shell"><video id="comparison-video" controls playsinline preload="metadata" poster="assets/quality-poster.jpg" aria-label="真实视频与三个模型的优化后同步对比"><source src="assets/model-comparison.mp4" type="video/mp4">请下载 MP4 观看。</video></div>
<div class="scene-picker" aria-label="跳转到场景"><button class="scene active" data-time="0"><img src="assets/quality-poster.jpg" alt="白天路口"><span>01 白天路口<small>0000 · 0:00</small></span></button><button class="scene" data-time="6.05"><img src="assets/quality-poster-1.jpg" alt="夜间街道"><span>02 夜间街道<small>0057 · 0:06</small></span></button><button class="scene" data-time="12.1"><img src="assets/quality-poster-2.jpg" alt="树荫人行道"><span>03 树荫人行道<small>0114 · 0:12</small></span></button></div>
<p class="caption-note">18.15 秒合辑；121 帧/片段，统一按 20 FPS 播放。播放速度与模型生成吞吐不同。视频为压缩观看副本，质量指标使用未压缩 RGB。</p>
<h3 class="result-title">Quantitative / 172 条验证质量</h3>@@QUALITY@@
<p class="caption-note">参考为原始相机 RGB。旧模型质量评测使用 RTX A5000；新模型使用 RTX A5000 / L40S 分片。本表不是分别在 L40S 和 A5000 上重跑两套 172 条质量评测；GPU 型号不代表不同模型版本。</p>
<h3 class="result-title">Throughput / 两种 GPU，统一优化后的单卡推理</h3><div class="table-wrap"><table class="hardware-table"><thead><tr><th scope="col">模型</th><th scope="col">NVIDIA L40S<br><small>48 GB · 单卡</small></th><th scope="col">NVIDIA RTX A5000<br><small>24 GB · 单卡</small></th></tr></thead><tbody>@@FPSROWS@@</tbody></table></div>
<div class="hardware-notes"><p><b>L40S：</b><span id="l40s-status">@@L40SSTATUS@@</span></p><p><b>A5000：</b><span id="a5000-status">@@A5000STATUS@@</span></p></div>
<div class="timing-box"><span class="small-label">统一计时口径 / 所有模型、所有 GPU</span><h4>不含加载、首帧编码、写文件。</h4><p>包含 rollout 生成＋TinyVAE BF16 流式解码；排除模型加载、cached 首帧编码、CPU 转换、MP4 写入和质量指标。固定视频 0000、0057、0114，各预热 1 次、计时 3 次，GPU 同步后计时。FPS = 360 个新帧 ÷ 三条视频各自中位耗时之和。</p><p>文本缓存在每次请求内重建，缓存建立成本包含在计时中；没有跨视频复用已建好的文本缓存。首帧 latent 已缓存，因此这不是从原始输入图到落盘视频的完整端到端延迟。</p></div>
<details class="detail"><summary>评测协议与可比性说明 <span>+</span></summary><div class="detail-body">@@PROTOCOL@@</div></details>
</section>
<section id="findings" class="log-section findings-section"><div class="section-head"><div><div class="eyebrow">03 / FINDINGS</div><h2>当前发现</h2></div></div>
<ol class="finding-list"><li><h3>4222 clean 的 FVD 点估计最低。</h3><p>FVD121 为 202.08，比旧 stage3 student 的 231.96 低 29.88（约 12.9%）。配对 bootstrap 95% 差值区间为 [−54.83, −9.27]。LPIPS、PSNR、SSIM 与旧模型接近，FVD 排名不能代表全部视觉维度。</p></li>
<li><h3>4222+noisy kv reuse 的配对画质指标最好。</h3><p>LPIPS=0.4150、PSNR=16.15 dB、SSIM=0.5142，在当前三模型中均值最好。FVD121=215.35，相对旧模型的差值区间 [−37.87, +1.82] 跨 0。</p></li>
<li><h3>更少的模型调用，已转化为实际吞吐收益。</h3><p>每个 6 秒 clip 的模型调用从旧模型 51 次降至 clean 的 33 次、noisy 的 24 次。两种 GPU 的实际 FPS 见上表；不按调用数比例推算硬件速度。</p></li>
<li><h3>这轮比较仍有需要拆开的变量。</h3><p>两版新模型训练长度不同；noisy 同时加入历史噪声匹配和 score 时间分布统一，并改变了 SP 规模。现有新 checkpoint 仍使用 121 行含 anchor 占位的动作约定，旧模型使用 120 行真实转移。不能将全部差异单独归因于 4222 或 noisy KV。</p></li></ol>
<div class="next-note"><span class="small-label">NEXT QUESTIONS</span><p>120 行动作约定重训；历史噪声匹配与 critic 时间抽样的独立消融；更长 rollout 与独立保留测试集。达到当前训练预算不等于已收敛。</p></div>
</section>
<footer class="entry-footer"><a href="results.json" download>指标 JSON ↓</a><a href="https://github.com/CeciliaTheBirb/research-log/tree/main/data/2026-10-25" target="_blank" rel="noopener">数据与复现记录 ↗</a><span>2026.10.25 · Cecilia</span></footer></main></div></body></html>'''
l40s = s['hardware_benchmarks']['l40s']
a5000 = s['hardware_benchmarks']['a5000']
lstatus = f'已完成，作业 {l40s["job_id"]}；三模型均为直接单卡＋文本缓存。'
astatus = (f'已完成，作业 {a5000["job_id"]}；同片段、同口径、同优化设置。' if a5000['status'] == 'completed'
           else f'作业 {a5000["job_id"]} 已进入正常 Slurm 队列，等待本轮评测完成；尚无优化后的实测值。')
for key, value in {'TRAINING': training, 'QUALITY': quality, 'FPSROWS': ''.join(fps_rows), 'PROTOCOL': protocol,
                   'L40SSTATUS': lstatus, 'A5000STATUS': astatus}.items():
    page = page.replace('@@' + key + '@@', value)
assert '@@' not in page
for forbidden in ['推理优化前后', 'speed_ablation', 'Clean3200', 'Noisy1600', 'localhost']:
    assert forbidden not in page, forbidden
(r / 'docs/index.html').write_text(page)
print('Built docs/index.html')
