"""Shared navigation and the independent October 27 experiment page."""
import html


def with_date_navigation(page, date):
    entries = [('2026-10-29', '2026-10-29.html', '10.29 · Edge × Nano 推理'),
               ('2026-10-27', '2026-10-27.html', '10.27 · Noisy KV + GAN'),
               ('2026-10-25', 'index.html', '10.25 · Stage 3 蒸馏')]
    dates = ''.join(
        f'<a class="date-link{" active" if day == date else ""}" href="{url}"'
        + (' aria-current="page"' if day == date else '')
        + f'><span aria-hidden="true">▤</span><span>{label}</span></a>'
        for day, url, label in entries)
    sections = ([('results', '同步对比视频'), ('rain-results', '雨后六视图'), ('quality-results', '质量指标'),
                 ('throughput-results', '推理速度'), ('history-feedback', '历史反馈对照'),
                 ('protocol', '配置与检查'), ('findings', 'Findings'), ('model-routing', '模型切换调研')]
                if date == '2026-10-29' else
                [('gan-results', '定性与定量结果'), ('throughput-results', '推理速度'),
                 ('noisy-gan', '训练细节'), ('findings', 'Findings'), ('checkpoints', 'Checkpoint')]
                if date == '2026-10-27' else
                [('model-design', '模型设计与训练细节'), ('results', '定性与定量结果'),
                 ('rain-results', '雨后路口'), ('ttur-results', 'Noisy KV · 1:5 实验'),
                 ('quality-results', '质量指标'), ('throughput-results', '推理速度'),
                 ('findings', 'Findings'), ('checkpoints', 'Checkpoint')])
    section_links = ''.join(f'<a href="#{key}">{label}</a>' for key, label in sections)
    sidebar = ('<aside class="date-sidebar" id="research-sidebar">'
               '<a href="index.html" class="workspace-name"><span class="workspace-icon">C</span><strong>cecilia-research-log</strong><span class="chevron" aria-hidden="true">⌄</span></a>'
               '<div class="workspace-caption">Research workspace</div>'
               '<a class="sidebar-home" href="index.html"><span aria-hidden="true">⌂</span>研究日志</a>'
               '<div class="sidebar-label">2026 · October</div><nav aria-label="按日期浏览">'
               + dates + '</nav><nav class="entry-nav" aria-label="本页目录">' + section_links
               + '</nav><div class="sidebar-bottom"><a href="https://github.com/CeciliaTheBirb/research-log" target="_blank" rel="noopener">GitHub repository ↗</a><span>cecilia · research notes</span></div></aside>')
    start, end = page.index('<aside '), page.index('</aside>') + len('</aside>')
    return page[:start] + sidebar + page[end:]


def build_gan_page(base_page, gan, training, results_block, results, checkpoints):
    assert gan.get('quality'), 'The dated results page requires completed quality evaluation.'
    speeds = gan['fps']
    names = [('old4444_native', '旧 stage3 student'),
             ('noisy_matched1600_nocaption', '4222+noisy kv reuse · 1:1基线'),
             ('noisy_gan1600_nocaption', '4222+noisy kv reuse + video GAN')]
    speed_rows = []
    for key, label in names:
        l40s = f'{speeds["l40s"][key]:.2f}<small>FPS</small>' if speeds.get('l40s') else '<span class="pending">测量中<small>Slurm 83085</small></span>'
        speed_rows.append(f'<tr><th scope="row">{label}</th><td>{l40s}</td><td>{speeds["a5000"][key]:.2f}<small>FPS</small></td></tr>')
    l40s_note = ('L40S作业83085已完成。' if speeds.get('l40s') else 'L40S作业83085已获分配并开始运行，完成后补入实测值。')
    speed_section = ('<section id="throughput-results" class="log-section"><h2>推理速度 / 同作业对照</h2>'
                     '<div class="table-wrap"><table class="hardware-table"><thead><tr><th scope="col">模型</th><th scope="col">NVIDIA L40S<small>48 GB · 单卡</small></th><th scope="col">NVIDIA RTX A5000<small>24 GB · 单卡</small></th></tr></thead><tbody>'
                     + ''.join(speed_rows) + '</tbody></table></div>'
                     '<p class="caption-note">A5000来自同一作业83084；' + l40s_note + '本页使用该轮次重新测速的基线，10.25页保留原轮次结果。两种GPU各自测量，不按硬件比例推算。</p>'
                     '<div class="timing-box"><span class="small-label">统一计时口径</span><h4>不含加载、首帧编码、写文件。</h4>'
                     '<p>直接单卡＋请求内文本缓存＋TinyVAE BF16流式。包含文本缓存建立、rollout生成与解码；排除模型加载、cached首帧编码、CPU转换、MP4写入及质量指标。0000、0057、0114各预热1次、计时3次；FPS = 360个新帧 ÷ 三条视频各自中位耗时之和。GAN判别器不进入推理。</p></div></section>')
    quality_rows = []
    for model_id in ['old', 'noisy', 'noisy_gan']:
        m = next(m for m in results['models'] if m['id'] == model_id)
        quality_rows.append(f'<tr><th scope="row">{html.escape(m["name"])}</th><td>{m["fvd121"]:.2f}</td><td>{m["fvd16"]:.2f}</td><td>{m["lpips"]:.4f}</td><td>{m["psnr"]:.2f}</td><td>{m["ssim"]:.4f}</td></tr>')
    old_comparison = ('<details class="detail"><summary>含旧学生的172条质量对照</summary><div class="detail-body"><div class="table-wrap"><table class="results-table"><thead><tr><th scope="col">模型</th><th scope="col">FVD121 ↓</th><th scope="col">FVD16 ↓</th><th scope="col">LPIPS ↓</th><th scope="col">PSNR ↑ dB</th><th scope="col">SSIM ↑</th></tr></thead><tbody>'
                      + ''.join(quality_rows) + '</tbody></table></div><p class="caption-note">新增GAN质量评测使用RTX A5000；基线质量沿用同一172条验证视频的既有结果。各模型共享canonical参考；MP4不参与指标计算。</p></div></details>')
    checkpoint_rows = []
    for c in checkpoints['models']:
        if c['name'] == '4222+noisy kv reuse' or c.get('id') == 'noisy_gan':
            checkpoint_rows.append(f'<tr><th scope="row">{html.escape(c["name"])}<small>step {c["step"]} · noisy KV</small></th><td><a href="{c["checkpoint_url"]}" target="_blank" rel="noopener">打开 checkpoint ↗</a></td><td><a href="{c["weights_url"]}" target="_blank" rel="noopener">adapter.safetensors ↗</a></td></tr>')
    checkpoint_section = ('<section id="checkpoints" class="log-section"><h2>Checkpoint</h2><p>保留原1:1基线；新增GAN是从相同Stage 2独立重新训练的版本。</p><div class="table-wrap"><table class="checkpoint-table"><thead><tr><th scope="col">模型</th><th scope="col">训练checkpoint</th><th scope="col">推理权重</th></tr></thead><tbody>'
                          + ''.join(checkpoint_rows) + '</tbody></table></div><p class="caption-note">最终GAN checkpoint1600已核对Box文件和metadata，state内含判别器与优化器。推理使用匹配Cosmos3-Edge底座及student adapter；判别器不参与推理。Box保留原有访问权限。</p></section>')
    header = ('<header class="entry-header"><div class="page-icon" aria-hidden="true">▤</div><div class="entry-date">2026 年 10 月 27 日</div><h1>Noisy KV + 动作条件视频 GAN</h1>'
              '<p class="entry-deck">保留1:1 Noisy KV基线，单独加入小权重GAN监督。训练与172条质量评测已完成：配对RGB指标小幅改善，FVD收益尚不确定。</p>'
              '<div class="entry-tags"><span>800 student / 800 fake</span><span>λGAN = 0.001</span><span>172 validation</span><span>4×RTX A5000</span></div></header>')
    training_details = ('<details class="detail"><summary>初始化、Noisy KV与判别器设置</summary><div class="detail-body common-settings"><dl>'
                        '<div><dt>初始化</dt><dd>student从causal Stage 2 step12550重新初始化；fake score及冻结teacher从bidir step30000加载。未从旧学生或此前Noisy checkpoint续训。</dd></div>'
                        '<div><dt>Rollout / Noisy KV</dt><dd>第一生成块4步，后续各2步。anchor与首个生成块使用clean KV；后续复用最后一步σ=0.625输入产生的KV。完整121帧仍为24次模型调用。</dd></div>'
                        '<div><dt>基线的两处匹配修改</dt><dd>辅助forward目标的后续GT历史在σ=0.625加噪，并匹配source/target时间；fake与DMD均采用shifted uniform时间抽样，shift=5，分别独立抽样。其余AnyFlow forward与DMD设置保持1:1基线。</dd></div>'
                        '<div><dt>Student / fake</dt><dd>AdamW学习率2×10⁻⁶，seed=0；4×A5000，SP4/DP1，有效batch=1。student与fake各800次；跨块历史KV detach。达到预算停止，不使用FVD early stopping。</dd></div>'
                        '<div><dt>GAN判别器</dt><dd>独立FP32时序CNN，width=32；输入归一化Cosmos视频latent、首帧及动作。丢弃anchor占位动作row0，将四次RGB转移对齐到一个未来latent时间。使用logistic GAN，AdamW lr=10⁻⁴、β=(0,0.99)，grad clip=1。</dd></div>'
                        '<div><dt>正则与数据</dt><dd>每16次判别器更新执行gamma=1的lazy R1；real/fake采用相同BF16量化与std=0.01 instance noise。GAN随机数流独立。4775条训练clip与172条验证在clip/session层面无重叠；训练用cached latent，无VAE解码。</dd></div>'
                        '<div><dt>更新与保存</dt><dd>student通过冻结D的输入梯度获得GAN监督；fake阶段使用detach样本更新D。D与fake score是两个网络。保存D权重、优化器及student/fake训练状态；单卡精确恢复与四卡梯度同步检查已通过。</dd></div>'
                        '</dl></div></details>')
    findings = ('<section id="findings" class="log-section findings-section"><h2>Findings</h2><ol class="finding-list">'
                '<li><h3>小权重GAN带来小幅配对画质收益。</h3><p>LPIPS、PSNR、SSIM的配对差值95%区间均不跨0。FVD121下降2.71，但区间[−14.62,+7.24]跨0，尚不能称为显著FVD提升。</p></li>'
                '<li><h3>保持1:1，推理速度基本相同。</h3><p>本轮没有增加fake score更新次数；GAN判别器随fake阶段单独更新。A5000同作业基线25.12、+GAN25.11 FPS；判别器不进入推理。</p></li>'
                '<li><h3>还需要独立测试与更长rollout。</h3><p>这是同一172条validation上的重复比较，训练loss后段仍有波动。这版是随机初始化的latent CNN GAN，不能将结果等同于完整官方DMD2。</p></li></ol></section>')
    footer = ('<footer class="entry-footer"><a href="results.json" download>指标 JSON ↓</a><a href="https://github.com/CeciliaTheBirb/research-log/tree/main/data/2026-10-26/noisy-conditional-gan" target="_blank" rel="noopener">数据、代码与复现记录 ↗</a><span>2026.10.27 · Cecilia</span></footer>')
    body = header + '<section id="results" class="log-section">' + results_block + old_comparison + '</section>' + speed_section + '<section id="model-design" class="log-section"><h2>模型设计与训练细节</h2>' + training + training_details + '</section>' + findings + checkpoint_section + footer
    main_start = base_page.index('<main class="log-content"')
    main_end = base_page.index('</main>', main_start) + len('</main>')
    page = base_page[:main_start] + '<main class="log-content" id="log-2026-10-27">' + body + '</main>' + base_page[main_end:]
    page = page.replace('href="#log-2026-10-25"', 'href="#log-2026-10-27"')
    page = page.replace('<span>10.25</span>', '<span>10.27</span>')
    page = page.replace('<title>cecilia-research-log · 2026.10.25</title>', '<title>cecilia-research-log · 2026.10.27 · Noisy KV + GAN</title>')
    return with_date_navigation(page, '2026-10-27')
