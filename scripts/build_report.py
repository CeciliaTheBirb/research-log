"""Build the date-indexed research log using only repository-local assets/data."""
from pathlib import Path
import html
import json

r = Path(__file__).resolve().parents[1]
s = json.loads((r / 'docs/results.json').read_text())
training = (r / 'scripts/training-content.html').read_text()
quality = (r / 'scripts/quality-table.html').read_text()
protocol = (r / 'scripts/protocol-content.html').read_text()
checkpoints = json.loads((r / 'docs/checkpoints.json').read_text())
checkpoint_rows = []
for c in checkpoints['models']:
    if c.get('id') == 'noisy_gan':
        continue
    checkpoint_rows.append(f'''<tr><th scope="row">{html.escape(c['name'])}<small>step {c['step']} · {html.escape(c['cache_mode'])} KV</small></th><td><a href="{c['checkpoint_url']}" target="_blank" rel="noopener">打开 checkpoint ↗</a></td><td><a href="{c['weights_url']}" target="_blank" rel="noopener">adapter.safetensors ↗</a><small>1.067 GB</small></td></tr>''')
fps_rows = []
for m in s['models']:
    if m['id'] == 'noisy_gan':
        continue
    cells = []
    for gpu in ['l40s', 'a5000']:
        b = s['hardware_benchmarks'][gpu]
        record = b['models'][m['id']]
        v = record['fps']
        pending_label = '测量中' if record.get('status') == 'running' else '排队中'
        cells.append(f'<td>{v:.2f}<small>FPS</small></td>' if v is not None else f'<td class="pending">{pending_label}<small>Slurm {record["job_id"]}</small></td>')
    fps_rows.append('<tr><th scope="row">' + html.escape(m['name']) + '</th>' + ''.join(cells) + '</tr>')
page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cecilia-research-log · 2026.10.25</title>
<meta name="description" content="Cecilia 的研究日志：模型设计与训练细节、定性与定量结果、Findings。Stage 3 蒸馏，L40S / A5000 优化后推理结果。">
<meta name="theme-color" content="#ffffff"><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="style.css?v=20261027"><script defer src="app.js?v=20261027"></script></head>
<body class="research-log"><a class="skip" href="#log-2026-10-25">跳到日志正文</a>
<aside class="date-sidebar" id="research-sidebar"><a href="#log-2026-10-25" class="workspace-name"><span class="workspace-icon">C</span><strong>cecilia-research-log</strong><span class="chevron" aria-hidden="true">⌄</span></a><div class="workspace-caption">Research workspace</div><a class="sidebar-home" href="#log-2026-10-25"><span aria-hidden="true">⌂</span>研究日志</a><div class="sidebar-label">2026 · October</div><nav aria-label="按日期浏览"><a class="date-link active" href="#log-2026-10-25" aria-current="page"><span aria-hidden="true">▤</span><span>10.25 · Stage 3 蒸馏</span></a></nav><nav class="entry-nav" aria-label="本页目录"><a href="#model-design">模型设计与训练细节</a><a href="#results">定性与定量结果</a><a href="#rain-results">雨后路口</a><a href="#quality-results">质量指标</a><a href="#throughput-results">推理速度</a><a href="#findings">Findings</a><a href="#checkpoints">Checkpoint</a></nav><div class="sidebar-bottom"><a href="https://github.com/CeciliaTheBirb/research-log" target="_blank" rel="noopener">GitHub repository ↗</a><span>cecilia · research notes</span></div></aside><div class="page-workspace"><header class="log-masthead"><div class="breadcrumb"><button class="sidebar-toggle" aria-label="切换侧栏" aria-controls="research-sidebar" aria-expanded="true">☰</button><span>Research Notes</span><span class="breadcrumb-slash">/</span><span>2026</span><span class="breadcrumb-slash">/</span><span>10.25</span></div><a href="https://github.com/CeciliaTheBirb/research-log" target="_blank" rel="noopener">在 GitHub 查看 ↗</a></header><main class="log-content" id="log-2026-10-25"><header class="entry-header"><div class="page-icon" aria-hidden="true">▤</div><div class="entry-date">2026 年 10 月 25 日</div><h1>Stage 3 蒸馏</h1><p class="entry-deck">从 Stage 2 初始化，训练 4222 clean 与 4222+noisy kv reuse。记录模型设计、质量与速度结果，以及当前发现。</p><div class="entry-tags"><span>Cosmos3-Edge</span><span>4775 train / 172 validation</span><span>TinyVAE BF16 流式</span></div></header>
<section id="model-design" class="log-section">@@TRAINING@@</section>
<section id="results" class="log-section"><div class="section-head"><div><div class="eyebrow">02 / QUALITATIVE + QUANTITATIVE RESULTS</div><h2>定性与定量结果</h2></div></div>
<p class="section-intro">速度表统一使用优化后的部署设置：旧 stage3 student 的原生直接单卡＋文本缓存，以及两版新模型的直接单卡＋文本缓存。模型历史 KV 的 clean / noisy 策略按各自训练方案保留。</p>
<h3 class="result-title">Qualitative / 同步对比视频</h3><div class="viewer-head"><p>上排：真实视频、旧 stage3 student<br>下排：4222 clean、4222+noisy kv reuse</p><a href="assets/model-comparison.mp4" download>下载 MP4 ↓</a></div>
<div class="video-shell"><video id="comparison-video" controls playsinline preload="metadata" poster="assets/quality-poster.jpg" aria-label="真实视频与三个模型的优化后同步对比"><source src="assets/model-comparison.mp4" type="video/mp4">请下载 MP4 观看。</video></div>
<div class="scene-picker" aria-label="跳转到场景"><button class="scene active" data-time="0"><img src="assets/quality-poster.jpg" alt="白天路口"><span>01 白天路口<small>0000 · 0:00</small></span></button><button class="scene" data-time="6.05"><img src="assets/quality-poster-1.jpg" alt="夜间街道"><span>02 夜间街道<small>0057 · 0:06</small></span></button><button class="scene" data-time="12.1"><img src="assets/quality-poster-2.jpg" alt="树荫人行道"><span>03 树荫人行道<small>0114 · 0:12</small></span></button></div>
<p class="caption-note">18.15 秒合辑；121 帧/片段，统一按 20 FPS 播放。播放速度与模型生成吞吐不同。视频为压缩观看副本，质量指标使用未压缩 RGB。</p>
<div id="rain-results" class="rain-results"><h3 class="result-title">雨后路口 / 三模型对比</h3><p>NavWorldGym 的 <code>rain_obs.png</code>，同一输入图与初始噪声，分别比较直行、慢行和静止动作。三个面板从左到右为旧 stage3 student、4222 clean、4222+noisy kv reuse。</p><div class="rain-input"><img src="assets/rain-input.png" alt="雨后路口输入首帧：左侧石墙、地面积水、前方路口"><div><strong>雨后路口 · 输入首帧</strong><span>seed 2026 · 121 帧 · 20 FPS · TinyVAE BF16 流式</span></div></div><div class="rain-controls" role="group" aria-label="选择雨后场景动作"><button class="active" data-rain="2p5mps" aria-pressed="true">直行 2.5 m/s</button><button data-rain="0p5mps" aria-pressed="false">慢行 0.5 m/s</button><button data-rain="stop" aria-pressed="false">静止</button></div><div class="video-shell"><video id="rain-video" controls playsinline preload="metadata" poster="assets/rain-poster-2p5mps.jpg" aria-label="雨后路口三模型同步对比"><source src="assets/rain-comparison-2p5mps.mp4" type="video/mp4"></video></div><div class="viewer-head rain-download"><span id="rain-action-label">直行 2.5 m/s · 三模型同步播放</span><a id="rain-download" href="assets/rain-comparison-2p5mps.mp4" download>下载当前 MP4 ↓</a></div><p class="caption-note">这是单张输入图的动作条件诊断，没有对应的真实未来视频。此展示样例不参与下方 172 条验证视频的指标统计；验证集 FVD、LPIPS、PSNR、SSIM 和 GPU 速度表保持原有结果。</p></div>
<h3 class="result-title" id="quality-results">Quantitative / 172 条验证质量</h3>@@QUALITY@@
<p class="caption-note">参考为原始相机 RGB。旧模型质量评测使用 RTX A5000；新模型使用 RTX A5000 / L40S 分片。本表不是分别在 L40S 和 A5000 上重跑两套 172 条质量评测；GPU 型号不代表不同模型版本。</p>
<h3 class="result-title" id="throughput-results">Throughput / 两种 GPU，统一优化后的单卡推理</h3><div class="table-wrap"><table class="hardware-table"><thead><tr><th scope="col">模型</th><th scope="col">NVIDIA L40S<br><small>48 GB · 单卡</small></th><th scope="col">NVIDIA RTX A5000<br><small>24 GB · 单卡</small></th></tr></thead><tbody>@@FPSROWS@@</tbody></table></div>
<div class="hardware-notes"><p><b>L40S：</b><span id="l40s-status">@@L40SSTATUS@@</span></p><p><b>A5000：</b><span id="a5000-status">@@A5000STATUS@@</span></p></div>
<div class="timing-box"><span class="small-label">统一计时口径 / 所有模型、所有 GPU</span><h4>不含加载、首帧编码、写文件。</h4><p>包含 rollout 生成＋TinyVAE BF16 流式解码；排除模型加载、cached 首帧编码、CPU 转换、MP4 写入和质量指标。固定视频 0000、0057、0114，各预热 1 次、计时 3 次，GPU 同步后计时。FPS = 360 个新帧 ÷ 三条视频各自中位耗时之和。</p><p>文本缓存在每次请求内重建，缓存建立成本包含在计时中；没有跨视频复用已建好的文本缓存。首帧 latent 已缓存，因此这不是从原始输入图到落盘视频的完整端到端延迟。</p></div>
<details class="detail"><summary>评测协议与可比性说明 <span>+</span></summary><div class="detail-body">@@PROTOCOL@@</div></details>
</section>
<section id="findings" class="log-section findings-section"><div class="section-head"><div><div class="eyebrow">03 / FINDINGS</div><h2>当前发现</h2></div></div>
<ol class="finding-list"><li><h3>旧学生后半段画质下降与撞墙，新模型有所改善。</h3><p>在本页对比样例中，尤其是雨后路口，旧 stage3 student 随 rollout 推进，后半段画质下降，并出现镜头撞向墙体的现象。新训练的 4222 clean 与 4222+noisy kv reuse 后半段画面相对稳定，撞墙问题也有所缓解。这是当前展示片段中的定性观察。</p></li>
<li><h3>4222 clean 的 FVD 点估计最低。</h3><p>FVD121 为 202.08，比旧 stage3 student 的 231.96 低 29.88（约 12.9%）。配对 bootstrap 95% 差值区间为 [−54.83, −9.27]。LPIPS、PSNR、SSIM 与旧模型接近，FVD 排名不能代表全部视觉维度。</p></li>
<li><h3>4222+noisy kv reuse 的配对画质指标最好。</h3><p>LPIPS=0.4150、PSNR=16.15 dB、SSIM=0.5142，在当前三模型中均值最好。FVD121=215.35，相对旧模型的差值区间 [−37.87, +1.82] 跨 0。</p></li>
<li><h3>更少的模型调用，已转化为实际吞吐收益。</h3><p>每个 6 秒 clip 的模型调用从旧模型 51 次降至 clean 的 33 次、noisy 的 24 次。两种 GPU 的实际 FPS 见上表；不按调用数比例推算硬件速度。</p></li>
<li><h3>这轮比较仍有需要拆开的变量。</h3><p>两版新模型训练长度不同；noisy 同时加入历史噪声匹配和 score 时间分布统一，并改变了 SP 规模。现有新 checkpoint 仍使用 121 行含 anchor 占位的动作约定，旧模型使用 120 行真实转移。不能将全部差异单独归因于 4222 或 noisy KV。</p></li></ol>
<div class="next-note"><span class="small-label">NEXT QUESTIONS</span><p>120 行动作约定重训；历史噪声匹配与 critic 时间抽样的独立消融；更长 rollout 与独立保留测试集。达到当前训练预算不等于已收敛。</p></div>
</section>
<section id="checkpoints" class="log-section"><div class="section-head"><h2>Checkpoint</h2></div><p>本页评测所用的两个训练版本。Box 文件与 <code>meta.json</code> 中的 step、Edge 架构、Stage 2 初始化及 clean / noisy 策略已核对。</p><div class="table-wrap"><table class="checkpoint-table"><thead><tr><th scope="col">模型</th><th scope="col">训练 checkpoint</th><th scope="col">推理权重</th></tr></thead><tbody>@@CHECKPOINTROWS@@</tbody></table></div><p class="caption-note">文件为 LoRA adapter，需配合对应 Cosmos3-Edge 底座和 Stage 2 初始化加载，推理时使用 student adapter。Checkpoint 目录同时包含训练 state；Box 链接沿用原有访问权限。</p></section><footer class="entry-footer"><a href="results.json" download>指标 JSON ↓</a><a href="https://github.com/CeciliaTheBirb/research-log/tree/main/data/2026-10-25" target="_blank" rel="noopener">数据与复现记录 ↗</a><span>2026.10.25 · Cecilia</span></footer></main></div></body></html>'''
l40s = s['hardware_benchmarks']['l40s']
a5000 = s['hardware_benchmarks']['a5000']
lstatus = f'已完成，作业 {l40s["job_id"]}；三模型均为直接单卡＋文本缓存。'
astatus = (f'已完成，作业 {a5000["job_id"]}；同片段、同口径、同优化设置。' if a5000['status'] == 'completed'
           else f'作业 {a5000["job_id"]} 已进入正常 Slurm 队列，等待本轮评测完成；尚无优化后的实测值。')
if s.get('hybrid_evaluation'):
    hybrid = next(m for m in s['models'] if m['id'] == 'hybrid')
    training += (r / 'scripts/hybrid-content.html').read_text()
    source = s['hf_source']
    hf_url = f'https://huggingface.co/{source["repo"]}/tree/{source["revision"]}/edge-text-hybrid/step-2400'
    checkpoint_rows.append(f'<tr><th scope="row">43222+hybrid forcing<small>HF Edge · step2400</small></th><td><a href="{hf_url}" target="_blank" rel="noopener">固定 HF 版本 ↗</a></td><td><a href="{hf_url}" target="_blank" rel="noopener">LoRA / action / time / history ↗</a><small>四组权重均需加载</small></td></tr>')
    edits = {
        '记录模型设计、质量与速度结果，以及当前发现。':'并加入 HF 发布的 43222+hybrid forcing（Edge step2400）作比较。',
        '4775 train / 172 validation':'4222：4775 train / 对比：172 validation',
        '上排：真实视频、旧 stage3 student<br>下排：4222 clean、4222+noisy kv reuse':'上排：旧 stage3 student、4222 clean<br>下排：4222+noisy kv reuse、43222+hybrid forcing',
        '真实视频与三个模型的优化后同步对比':'四个模型的优化后同步对比',
        '雨后路口 / 三模型对比':'雨后路口 / 四模型对比',
        '三个面板从左到右为旧 stage3 student、4222 clean、4222+noisy kv reuse。':'上排为旧 stage3 student、4222 clean；下排为 4222+noisy kv reuse、43222+hybrid forcing。',
        '雨后路口三模型同步对比':'雨后路口四模型同步对比',
        '三模型同步播放':'四模型同步播放',
        '旧 stage3 student 的原生直接单卡＋文本缓存，以及两版新模型的直接单卡＋文本缓存。':'四个模型均采用直接单卡＋请求内文本缓存。',
        '验证集 FVD、LPIPS、PSNR、SSIM 和 GPU 速度表保持原有结果。':'质量表来自 172 条验证视频，速度表来自独立重复测速；本条雨天展示不改变这些统计口径。',
        '4222 clean 的 FVD 点估计最低。':'原三模型比较中，4222 clean 的 FVD 点估计最低。',
        '4222+noisy kv reuse 的配对画质指标最好。':'原三模型比较中，4222+noisy kv reuse 的配对画质指标最好。',
        '在当前三模型中均值最好':'在原三模型比较中均值最好',
        '旧模型质量评测使用 RTX A5000；新模型使用 RTX A5000 / L40S 分片。':'旧学生和 HF hybrid 使用 RTX A5000；两版 4222 使用 RTX A5000 / L40S 分片。',
        '本页评测所用的两个训练版本。':'本页两版 4222 的 Box checkpoint，以及新增的 HF hybrid 固定版本。',
        'Checkpoint 目录同时包含训练 state；Box 链接沿用原有访问权限。':'Box checkpoint 目录同时包含训练 state，链接沿用原有访问权限。HF hybrid 需 Cosmos3-Edge 底座及四组导出权重；HF 链接可能需要原仓库访问权限。',
    }
    for old, new in edits.items():
        assert old in page, old
        page = page.replace(old, new)
    page = page.replace('质量指标使用未压缩 RGB。</p>', '质量指标使用未压缩 RGB。<a href="assets/model-comparison-with-reference.mp4" download>原三模型＋真实参考视频 ↓</a></p>')
    h_fps = a5000['models']['hybrid']['fps']
    finding = (f'<li><h3>新增 HF hybrid 的完整 checkpoint 比较。</h3><p>172 条相同验证视频上，'
               f'FVD121={hybrid["fvd121"]:.2f}、FVD16={hybrid["fvd16"]:.2f}，LPIPS={hybrid["lpips"]:.4f}、'
               f'PSNR={hybrid["psnr"]:.2f} dB、SSIM={hybrid["ssim"]:.4f}；A5000 为 {h_fps:.2f} FPS。'
               'SSIM 均值为四模型最高，但 FVD 和 LPIPS 均差于原三模型。'
               '采用原生 4,3,2,2,… 与 sink1/history2＋历史记忆；这里使用空文本，官方示例使用结构化 prompt。其训练数据、窗口和初始化也不同，结果不构成记忆模块的独立消融。</p></li>')
    audit_path = r / 'data/2026-10-25/hf-hybrid/config-audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        assert audit['passed'] and audit['count'] == 10
        labels = [('native_s1h2_empty','默认窗口＋记忆，空文本'),
                  ('full_history_empty','完整历史，空文本'),
                  ('memory_disabled_empty','窗口＋关闭记忆，空文本'),
                  ('native_s1h2_structured_generic','窗口＋记忆，通用结构化文本')]
        audit_rows = []
        for key, label in labels:
            q = audit['variants'][key]
            audit_rows.append(f'<tr><th scope="row">{label}</th><td>{q["lpips_alex"]:.4f}</td><td>{q["psnr_db"]:.2f}</td><td>{q["ssim"]:.4f}</td></tr>')
        finding += ('<li id="hybrid-config-audit"><h3>Hybrid 的窗口与历史记忆已核查。</h3>'
                    '<p>sink1/history2、absolute RoPE、记忆读取和4322…步数与官方默认一致。默认路径重跑后RGB与原评测完全一致；抽查第0、13、27层，63次具有移出历史的读取均有非零贡献。关闭记忆后贡献为0。</p>'
                    '<div class="table-wrap"><table class="results-table"><caption>固定十条视频的诊断；不替换172条FVD</caption><thead><tr><th>设置</th><th>LPIPS ↓</th><th>PSNR ↑ dB</th><th>SSIM ↑</th></tr></thead><tbody>'
                    + ''.join(audit_rows) + '</tbody></table></div>'
                    '<p>结构化文本由官方minimal helper生成，仅包含通用车载前视街景信息。它不是完整场景prompt，不能据此认定空文本是退步根因。'
                    '<a href="https://github.com/CeciliaTheBirb/research-log/blob/main/data/2026-10-25/hf-hybrid/config-audit-zh.md" target="_blank" rel="noopener">配置、逐块窗口与读取核查记录 ↗</a></p></li>')
    page = page.replace('<ol class="finding-list">', '<ol class="finding-list">' + finding)
    if (r / 'data/2026-10-25/dmd2-noisy-kv-review.md').exists():
        page = page.replace('<div class="next-note">', '<p class="caption-note"><a href="https://github.com/CeciliaTheBirb/research-log/blob/main/data/2026-10-25/dmd2-noisy-kv-review.md" target="_blank" rel="noopener">DMD2 与 Noisy KV：原理、当前实现差异及建议实验 ↗</a></p><div class="next-note">')
    for label, benchmark in [('L40S', l40s), ('A5000', a5000)]:
        h = benchmark['models']['hybrid']
        state = f'hybrid 作业 {h["job_id"]} 已完成。' if h['fps'] is not None else f'hybrid 作业 {h["job_id"]} 排队中，尚无实测值。'
        text = f'原三模型作业 {benchmark["job_id"]}；' + state + ' 同片段、同计时口径，hybrid 为另一轮单卡作业。'
        if label == 'L40S': lstatus = text
        else: astatus = text
ttur_path = r / 'data/2026-10-25/noisy-ttur15/experiment.json'
if ttur_path.exists():
    ttur = json.loads(ttur_path.read_text())
    assert ttur['planned_updates'] == {'student': 800, 'fake': 4000, 'total': 4800}
    assert ttur['preflight']['reload_verified'] and ttur['preflight']['gradient_isolation_verified']
    jobs = '、'.join(str(j['job']) for j in ttur['training_jobs'])
    evaluation_jobs = '、'.join(str(j['job']) for j in ttur['evaluation_jobs'])
    ttur_l40s = s['hardware_benchmarks']['l40s']['models']['noisy_ttur15']['fps']
    ttur_l40s_note = 'L40S作业82762已完成。' if ttur_l40s is not None else 'L40S作业82762排队中。'
    completed = ttur.get('quality') is not None
    resources = '已使用' if completed else '使用'
    training_result = (
        f'<p>训练作业 {jobs} 已完成800次student＋4000次fake，总计4800次更新；两段完整恢复模型及优化器。'
        f'占卡13小时24分，约53.6 GPU小时（含加载、保存和同步）。172条质量与A5000评测作业82761已完成；{ttur_l40s_note}最终checkpoint4800已备份到Box，链接见页末。</p>'
        if completed else
        f'<p>训练作业：{jobs}。评测作业：{evaluation_jobs}；结果完成后更新。</p>'
    )
    training += ('<div class="model-card" id="noisy-ttur15"><h3>4222+noisy kv reuse · TTUR1:5</h3>'
                 f'<p><strong>{html.escape(ttur["status_zh"])}</strong>。状态记录时间：{html.escape(ttur["updated_at"])}。</p>'
                 '<p>从同一 Stage 2 重新初始化，保留4222、历史噪声匹配、score时间抽样、损失和学习率；student:fake从1:1改为1:5。本轮没有GAN。</p>'
                 f'<p>{resources}4张RTX A5000，SP4 / DP1，有效batch=1；预算800次student＋4000次fake，共4800次总更新。每50次student更新验证和保存。</p>'
                 + training_result +
                 '<p>预检查已完成2次student＋10次fake更新，梯度隔离及新进程重载一致性通过；训练4775条与验证172条在clip/session层面无重叠。</p>'
                 '<p>800次student预算与1:1基线相同；沿用loader与seed，但更新比例改变了进入student的样本及逐步随机数，不是逐student更新的完全配对训练。这里只测试更新频率，尚未加入完整DMD2的其他组件。</p>'
                 '<p><a href="https://github.com/CeciliaTheBirb/research-log/blob/main/data/2026-10-25/noisy-ttur15/README.md" target="_blank" rel="noopener">实验状态、训练配方与验证记录 ↗</a></p></div>')
    if completed:
        ttur_results = (
            '<div id="ttur-results" class="rain-results"><h3 class="result-title">Noisy KV / 1:1 与 1:5 更新比例</h3>'
            '<p><strong>本轮1:5未显示整体质量收益。</strong>相同800次student更新预算下，FVD121从215.35变为221.91，LPIPS、PSNR和SSIM也略差；FVD16从278.85降至275.87。'
            'FVD121差值为+6.56，配对bootstrap 95%区间[−7.98,+15.77]跨0，尚不能称为显著退步；这也不等于完整DMD2无效。</p>'
            '<p>配对RGB指标的差值区间则均不跨0：LPIPS为+0.01084 [0.00796,0.01377]，PSNR为−0.1756 dB [−0.2348,−0.1166]，SSIM为−0.00471 [−0.00637,−0.00304]。这172条验证视频上，1:5的配对画质指标确有下降。</p>'
            '<div class="viewer-head"><p>上排：真实视频、旧 stage3 student<br>下排：4222+noisy kv reuse（1:1）、TTUR1:5</p>'
            '<a href="assets/noisy-ttur15-comparison.mp4" download>下载 MP4 ↓</a></div>'
            '<div class="video-shell"><video id="ttur-video" controls playsinline preload="metadata" poster="assets/noisy-ttur15-comparison.jpg" '
            'aria-label="真实视频、旧学生及Noisy KV两种更新比例的同步对比"><source src="assets/noisy-ttur15-comparison.mp4" type="video/mp4">请下载MP4观看。</video></div>'
            '<p class="caption-note">三个场景：0000白天路口、0057夜间街道、0114树荫人行道；共18.15秒，20 FPS播放。172条质量评测已逐条核对首帧、动作、文本、seed与初始噪声；两版noisy均为24次模型调用。</p>'
            '<p>同一作业82761的A5000优化后实测：1:1为<strong>25.76 FPS</strong>，1:5为<strong>25.73 FPS</strong>，速度基本相同。'
            f'下方总表保留1:1旧轮次的26.19 FPS；判断本次速度差异应使用这组同作业对照。{ttur_l40s_note}</p></div>'
        )
        page = page.replace('<h3 class="result-title" id="quality-results">', ttur_results + '<h3 class="result-title" id="quality-results">')
        page = page.replace('<a href="#quality-results">质量指标</a>', '<a href="#ttur-results">Noisy KV · 1:5 实验</a><a href="#quality-results">质量指标</a>')
        page = page.replace('四个模型均采用直接单卡＋请求内文本缓存。', '各模型均采用直接单卡＋请求内文本缓存。')
        page = page.replace('本页两版 4222 的 Box checkpoint，以及新增的 HF hybrid 固定版本。', '本页三版 4222 的 Box checkpoint（含TTUR1:5），以及HF hybrid固定版本。')
        page = page.replace('两版 4222 使用 RTX A5000 / L40S 分片。', '原两版 4222 使用 RTX A5000 / L40S 分片，TTUR1:5使用RTX A5000。')
        finding = ('<li><h3>Noisy KV 的1:5更新频率未显示整体收益。</h3>'
                   '<p>相同800次student更新预算，FVD121为221.91（1:1为215.35），配对差值区间跨0；LPIPS、PSNR、SSIM略差，FVD16略好。'
                   '同作业A5000速度为25.76→25.73 FPS。暂保留1:1作为基线，不能仅凭增加fake更新次数认定critic滞后已经解决，或将结果推广到完整DMD2。</p></li>')
        page = page.replace('<ol class="finding-list">', '<ol class="finding-list">' + finding)
        astatus += ' TTUR1:5及1:1同作业对照82761已完成。'
        lstatus += ' ' + ttur_l40s_note
        page = page.replace('2026.10.25 · Cecilia</span>', '2026.10.26 更新 · Cecilia</span>')
gan_training = gan_results = ''
gan_path = r / 'data/2026-10-26/noisy-conditional-gan/experiment.json'
if gan_path.exists():
    gan = json.loads(gan_path.read_text())
    completed = gan.get('quality') is not None
    run_status = html.escape(gan['status_zh'])
    job_note = f'正式训练作业 {gan["training_job"]}；' if gan.get('training_job') else '正式训练尚未启动；'
    evaluation_note = '、'.join(str(j['job']) for j in gan.get('evaluation_jobs', []))
    if evaluation_note:
        job_note += f'评测作业{evaluation_note}。' if completed else f'训练成功后运行评测作业{evaluation_note}。'
    check_note = ('单卡训练与新进程精确恢复、22项单元/回归测试、四卡全分辨率前后向及R1正则检查均已通过；'
                  f'四卡预检rank0峰值allocated显存{gan["preflight"]["peak_full_resolution_gb_rank0"]:.2f} GiB。') if gan.get('preflight', {}).get('passed') else '正式训练前检查梯度隔离、动作对齐、同步与恢复。'
    progress_note = ''
    if gan.get('progress'):
        gp = gan['progress']
        progress_note = (f'<p>已完成{gp["student_updates"]}次student与{gp["fake_and_discriminator_updates"]}次fake/GAN判别器更新；checkpoint{gp["latest_checkpoint"]}的adapter、训练state及GAN优化器已核实备份到Box。所有记录值有限，rank0峰值allocated显存{gp["peak_allocated_gib"]:.2f} GiB。</p>')
        if completed:
            progress_note += ('<p>使用4×RTX A5000，SP4 / DP1、有效batch=1，正式训练耗时14小时06分28秒，约56.43 GPU小时。固定4个validation batch的辅助forward loss从0.21846变为0.21178，最低为step1200的0.21105，后段有波动，尚不能宣称收敛。'
                              'warmup后稀疏记录的加权GAN/DMD输出latent梯度RMS比为1.17%–12.02%；这是输出端诊断，不是完整参数梯度比例。</p>')
        else:
            progress_note += f'<p>按当前速度估计还需约{gp["remaining_hours_estimate"]:.1f}小时训练，排队与后续评测另计。</p>'
    conclusion_note = '172条质量与两种GPU重复测速已完成，结果及同作业基线对照见本页结果区。' if completed else '本轮尚无新FVD/FPS结论，不替换下方已有结果。'
    gan_training = ('<div class="model-card" id="noisy-gan"><h3>新实验 / 1:1 Noisy KV + 动作条件视频GAN</h3>'
                 f'<p><strong>{run_status}</strong>。记录时间：{html.escape(gan["updated_at"])}。{job_note}</p>'
                 f'{progress_note}'
                 '<p>保留1:1基线，从相同Stage2重新初始化；仍是800次student＋800次fake score。新增判别器随fake阶段更新800次，fake score与GAN判别器是两个独立网络。</p>'
                 '<p>student目标为AnyFlow forward＋2×DMD＋λ×GAN；λ在前50次student更新逐渐增至0.001。判别器读取视频latent、首帧和对齐的动作序列，学习时序与画面真实性；仅从4775条训练clip取真实样本。</p>'
                 '<p>这版使用独立、随机初始化的轻量时序CNN，在latent空间训练，不是DMD2官方的score特征判别器。判别器不会进入推理；评测仍用TinyVAE BF16流式、直接单卡＋文本缓存。</p>'
                 f'<p>{check_note}使用相同172条validation与初始噪声评测。{conclusion_note}</p>'
                 '<p><a href="https://github.com/CeciliaTheBirb/research-log/tree/main/data/2026-10-26/noisy-conditional-gan" target="_blank" rel="noopener">实验配方、代码与验证记录 ↗</a></p></div>')
    if completed:
        paired = gan['paired_comparison']
        quality_rows = ''
        baseline = next(m for m in s['models'] if m['id'] == 'noisy')
        candidate = next(m for m in s['models'] if m['id'] == 'noisy_gan')
        for label, key, spec in [('FVD121 ↓', 'fvd121', '.2f'), ('FVD16 ↓', 'fvd16', '.2f'), ('LPIPS ↓', 'lpips', '.4f'), ('PSNR ↑ dB', 'psnr', '.2f'), ('SSIM ↑', 'ssim', '.4f')]:
            quality_rows += f'<tr><th scope="row">{label}</th><td>{baseline[key]:{spec}}</td><td>{candidate[key]:{spec}}</td></tr>'
        same_job = gan['fps']['a5000']
        gan_results = (
            '<div id="gan-results" class="rain-results"><h3 class="result-title">1:1 Noisy KV + 小权重动作条件视频GAN</h3>'
            '<p><strong>配对RGB画质指标有小幅改善，FVD收益尚不确定。</strong>172条相同validation上，LPIPS、PSNR、SSIM均改善；FVD121从215.35变为212.64，差值−2.71，配对bootstrap 95%区间[−14.62,+7.24]跨0，不能称为显著FVD提升。</p>'
            '<div class="table-wrap"><table class="results-table gan-quality-table"><caption>相同800次student预算；新增GAN权重0.001</caption><thead><tr><th scope="col">指标</th><th scope="col">Noisy KV / 1:1基线</th><th scope="col">Noisy KV / 1:1 + GAN</th></tr></thead><tbody>' + quality_rows + '</tbody></table></div>'
            '<p>相对1:1基线的配对差值：LPIPS −0.00823 [−0.01029,−0.00625]；PSNR +0.1597 dB [0.1140,0.2054]；SSIM +0.00536 [0.00407,0.00663]。三项95%区间均不跨0。FVD121区间使用512次配对重采样；这是同一验证集上的比较，尚需独立测试和长rollout验证。</p>'
            '<div class="viewer-head"><p>上排：真实视频、旧 stage3 student<br>下排：Noisy KV 1:1基线、Noisy KV 1:1 + GAN</p><a href="assets/noisy-gan-comparison.mp4" download>下载 MP4 ↓</a></div>'
            '<div class="video-shell"><video id="gan-video" controls playsinline preload="metadata" poster="assets/noisy-gan-comparison.jpg" aria-label="真实视频、旧学生、Noisy KV基线与新增动作条件GAN的同步对比"><source src="assets/noisy-gan-comparison.mp4" type="video/mp4">请下载MP4观看。</video></div>'
            '<p class="caption-note">三个场景：0000白天路口、0057夜间街道、0114树荫人行道；共18.15秒，统一20 FPS播放。两版Noisy的172条记录已逐条核对clip、seed、初始噪声、文本hash及24次模型调用；同一输入数据与历史策略。指标使用未压缩RGB，MP4仅供观看。</p>'
            f'<p>同一作业83084、同一张RTX A5000：1:1基线<strong>{same_job["noisy_matched1600_nocaption"]:.2f} FPS</strong>，+GAN<strong>{same_job["noisy_gan1600_nocaption"]:.2f} FPS</strong>，旧学生{same_job["old4444_native"]:.2f} FPS。GAN判别器不进入推理，速度基本相同。'
            '均为直接单卡＋请求内文本缓存＋TinyVAE BF16流式，不含加载、首帧编码与写文件。两种GPU的同轮次对照见下方速度表。</p>'
            '<p><a href="https://github.com/CeciliaTheBirb/research-log/blob/main/data/2026-10-26/noisy-conditional-gan/RESULTS.md" target="_blank" rel="noopener">完整配对统计、逐条指标与测速记录 ↗</a></p></div>'
        )
for key, value in {'TRAINING': training, 'QUALITY': quality, 'FPSROWS': ''.join(fps_rows), 'PROTOCOL': protocol,
                   'L40SSTATUS': lstatus, 'A5000STATUS': astatus, 'CHECKPOINTROWS': ''.join(checkpoint_rows)}.items():
    page = page.replace('@@' + key + '@@', value)
assert '@@' not in page
for forbidden in ['推理优化前后', 'speed_ablation', 'Clean3200', 'Noisy1600', 'localhost', '4222 具体省掉了什么', 'block-track']:
    assert forbidden not in page, forbidden
from report_dates import build_gan_page, with_date_navigation
page = with_date_navigation(page, '2026-10-25')
(r / 'docs/index.html').write_text(page)
if gan_results:
    new_page = build_gan_page(page, gan, gan_training, gan_results, s, checkpoints)
    assert '@@' not in new_page
    (r / 'docs/2026-10-27.html').write_text(new_page)
print('Built date-indexed research pages')
