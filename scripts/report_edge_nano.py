"""Build the trained-student inference page from explicit result snapshots."""
import html
import json

from report_dates import with_date_navigation


def build_edge_nano_page(base_page, data):
    result = json.loads((data / 'interim.json').read_text())
    bench = json.loads((data / 'benchmark.json').read_text())
    protocol = json.loads((data / 'protocol.json').read_text())
    count = result['matched_count']
    assert count == len(result['matched_indices'])
    assert result['all_noise_actions_matched'] and result['all_shadow_final_latents_equal']
    assert bench['gpu_count'] == 3 and bench['timed_repeats'] == 3
    labels = [('edge4444', 'EEEE · 纯 Edge'), ('nano4444', 'NNNN · 纯 Nano'),
              ('nano_first', 'NEEE · Nano 开头'), ('nano_last', 'EEEN · Nano 收尾')]
    quality_rows = []
    speed_rows = []
    for key, label in labels:
        q = result['variants'][key]
        quality_rows.append(f'<tr><th scope="row">{label}</th><td>{q["lpips_alex"]:.4f}</td>'
                            f'<td>{q["psnr_db"]:.2f}</td><td>{q["ssim"]:.4f}</td><td>待全量</td></tr>')
        fps = bench['variants'][key]['fps_from_median_seconds']
        speed_rows.append(f'<tr><th scope="row">{label}</th><td>{fps:.2f}<small>FPS</small></td><td>排队中</td></tr>')
    videos = []
    for index, title in [(0, '白天路口'), (57, '夜间街道'), (114, '树荫人行道')]:
        stem = f'assets/edge-nano-20261029/comparison-{index:04d}'
        videos.append(f'<h3 class="result-title">{index:04d} · {title}</h3>'
                      f'<div class="video-shell"><video controls playsinline preload="metadata" poster="{stem}.jpg" '
                      f'aria-label="{title}：真实参考与Edge Nano五种推理方案的同步对比">'
                      f'<source src="{stem}.mp4" type="video/mp4">请下载 MP4 观看。</video></div>'
                      f'<p class="caption-note"><a href="{stem}.mp4" download>下载这组 MP4 ↓</a> · 6.05 秒 / 20 FPS</p>')
    feedback_rows = []
    for key, label in [('nano_last_feedback_diagnostic', 'Edge 提交 Nano 最终 latent'),
                       ('nano_last_draft_history', 'Edge 提交自己的末步 draft')]:
        q = result['variants'][key]
        fps = bench['variants'][key]['fps_from_median_seconds']
        feedback_rows.append(f'<tr><th scope="row">{label}</th><td>{q["lpips_alex"]:.4f}</td>'
                             f'<td>{q["psnr_db"]:.2f}</td><td>{q["ssim"]:.4f}</td><td>{fps:.2f}</td></tr>')
    final = result['variants']['nano_last_feedback_diagnostic']['lpips_alex']
    draft = result['variants']['nano_last_draft_history']['lpips_alex']
    wins = result['final_history_vs_draft_lpips_wins']
    stamp = html.escape(protocol['observed_at'])
    repo = 'https://github.com/CeciliaTheBirb/research-log/tree/main/data/2026-10-29/edge-nano-4444'
    header = f'''<header class="entry-header"><div class="page-icon" aria-hidden="true">▤</div>
<div class="entry-date">2026 年 10 月 29 日</div><h1>Edge × Nano 混合推理</h1>
<p class="entry-deck">使用已经训练好的普通版四步学生，比较 Nano 做第一步、最后一步，以及它的修正能否通过 Edge 历史帮助后续生成。</p>
<div class="entry-tags"><span>Inference only</span><span>每 chunk 四步</span><span>3×RTX A5000</span><span>TinyVAE BF16 流式</span></div>
<div class="timing-box"><strong>全量评测运行中 · 已匹配 {count} / 172 条</strong>
<p>本页是阶段快照。FPS 已完成预热和三次重复测试；质量表仅汇总共同完成的 {count} 条，FVD 等待全部172条完成。L40S 仍排队，没有填入估算值。</p>
<p>状态记录：{stamp}（洛杉矶时间）。本页不会随服务器日志自动刷新。</p></div></header>'''
    results = '''<section id="results" class="log-section"><h2>同步对比视频</h2>
<p>E = Edge，N = Nano；字母顺序对应每个 chunk 从高噪声到低噪声的四个去噪区间。</p>
<p class="section-intro">上排：真实参考 / 纯 Edge / 纯 Nano。<br>下排：Nano 开头 / Nano 收尾并提交最终历史 / Nano 收尾但 Edge 保留 draft 历史。</p>'''
    results += ''.join(videos) + '<p class="caption-note">视频为压缩观看副本；质量指标来自压缩前 RGB。20 FPS 播放速度与模型推理吞吐不同。</p></section>'
    quality = f'''<section id="quality-results" class="log-section"><h2>质量指标 · 阶段结果</h2>
<p>同一组 {count} 条 validation、相同输入图、真实动作和 canonical 初始噪声。121 帧 / 480×832；配对 RGB 指标排除已知首帧。</p>
<div class="table-wrap"><table class="results-table"><caption>共同完成 {count}/172 条；并非最终结果</caption>
<thead><tr><th scope="col">每 chunk 路由</th><th scope="col">LPIPS ↓</th><th scope="col">PSNR ↑ dB</th><th scope="col">SSIM ↑</th><th scope="col">FVD121 / 16 ↓</th></tr></thead>
<tbody>{''.join(quality_rows)}</tbody></table></div>
<p class="caption-note">质量快照索引：{html.escape(', '.join(result['matched_indices']))}。这些是反复用于比较的 validation；外部学生历史训练数据与它的重叠尚未确认，不声称这是外部模型完全未见过的测试集。</p>
<p>真实参考视频和 I3D 特征已经准备好。FVD 待全量，是因为当前只完成部分样本；有参考数据的 PSNR、SSIM、LPIPS 已可逐条统计。</p></section>'''
    speed = f'''<section id="throughput-results" class="log-section"><h2>推理速度</h2>
<div class="table-wrap"><table class="hardware-table"><thead><tr><th scope="col">路由</th><th scope="col">3×RTX A5000<small>Edge 1 卡 / Nano 2 卡</small></th><th scope="col">1×L40S<small>作业84980</small></th></tr></thead><tbody>{''.join(speed_rows)}</tbody></table></div>
<div class="timing-box"><h4>包含模型切换、历史提交与 TinyVAE 解码。</h4>
<p>固定视频57，预热1次、计时3次。FPS = 120个新帧 ÷ 三次总耗时的中位数。计入文本缓存构造、去噪、设备传输、两个活跃模型的历史提交和 TinyVAE BF16 流式解码；排除模型加载、cached 首帧编码、CPU 转换、写文件与质量指标。</p>
<p>三张 A5000 为同一作业分配：Edge 常驻 CUDA0；Nano 前18层及输入/输出在 CUDA1，后18层在 CUDA2。纯模型路由只调用自身，另一模型闲置。这里的 FPS 是该分卡部署的吞吐，不能称作单张 A5000 结果或据此推断 GPU 成本更低。</p></div></section>'''
    feedback = f'''<section id="history-feedback" class="log-section"><h2>Nano 的修正能否帮助后续生成？</h2>
<p>两组都从第三步后的相同状态，计算 Edge 和 Nano 的末步，展示都用 Nano 输出。唯一变化：Edge 的历史写入自己的 draft，还是最终 Nano latent。Nano 自己始终写入最终输出。</p>
<div class="table-wrap"><table class="results-table"><thead><tr><th scope="col">Edge 历史内容</th><th scope="col">LPIPS ↓</th><th scope="col">PSNR ↑ dB</th><th scope="col">SSIM ↑</th><th scope="col">诊断 FPS</th></tr></thead><tbody>{''.join(feedback_rows)}</tbody></table></div>
<p>当前 {count} 条中，最终历史的 LPIPS 在 {wins} 条上更低；均值 {draft:.4f} → {final:.4f}。两组第一 chunk 逐值相同，后续才因历史而产生差异。这是初步趋势，尚不能认定稳定收益。</p>
<p class="caption-note">这里没有跨模型复制 KV：最终 latent 分别经过 Edge、Nano 的 clean commit，写入各自原生缓存。诊断包含额外一次 Edge 末步，速度不能冒充上方实际 EEEN 部署速度。</p></section>'''
    checks = '''<section id="protocol" class="log-section"><h2>配置与检查</h2>
<div class="detail-body common-settings"><dl>
<div><dt>学生权重</dt><dd>Edge：edge-anyflow-causal / step2200。Nano：nano-v0.1 / step1070，非 EMA。两者均为已经训练过的普通 causal AnyFlow student；不是此前撤下的原生 base 模型测试。</dd></div>
<div><dt>统一采样</dt><dd>每 chunk 四步，CFG1、shift5；实际时间网格 [1, 0.9375, 0.8333333, 0.625, 0]。相同 native action-format prompt，完整 clean 历史，(1,3) latent 分块，两套独立文本/KV缓存。</dd></div>
<div><dt>解码与数据</dt><dd>TinyVAE step38000，BF16 流式。相同 cached anchor、原始 RGB 参考、真实动作和逐 chunk 初始噪声；没有重训或修改权重。</dd></div>
<div><dt>原生一致性</dt><dd>新 runner 的纯 EEEE、NNNN 各自与原生生成逐值一致，最大差0。Nano 18/18 与20/16分层位置的两生成 chunk 输出也逐值一致。</dd></div>
<div><dt>反馈隔离</dt><dd>加入 Edge shadow 分支不改变最终历史结果；已完成共同样本均通过 latent hash 核对。两反馈组第一 chunk 一致，之后才出现历史造成的变化。</dd></div>
<div><dt>资源</dt><dd>Slurm84988，bolei-gpu02，3×A5000，12小时上限。启动前通过空闲 GPU 检查；没有终止其他用户进程。未使用量化、CPU offload、梯度或跨网络反传。</dd></div>
</dl></div></section>'''
    findings = '''<section id="findings" class="log-section findings-section"><h2>Findings · 暂定</h2><ol class="finding-list">
<li><h3>当前混合方案处在两种纯模型之间。</h3><p>共同完成样本的平均 LPIPS 均优于纯 Edge，但不及纯 Nano；混合 FPS 高于纯 Nano、低于纯 Edge。样本仍少，尚无最终 FVD 结论。</p></li>
<li><h3>开头与收尾各有作用，不能只凭模型大小选步骤。</h3><p>当前平均 LPIPS 是 Nano 开头更低，SSIM 是 Nano 收尾更高；具体场景仍有退化。不能从少量视频判断一种顺序全面更好。</p></li>
<li><h3>历史反馈有初步正向趋势。</h3><p>把 Nano 最终 latent 写入 Edge 后，当前平均配对指标略好。完整172条完成后再判断收益是否稳定，以及它在后续 chunk 能保持多久。</p></li>
</ol></section>'''
    footer = f'''<footer class="entry-footer"><a href="edge-nano-results.json" download>阶段质量 JSON ↓</a><a href="{repo}" target="_blank" rel="noopener">数据与测速记录 ↗</a><span>2026.10.29 · Cecilia</span></footer>'''
    body = header + results + quality + speed + feedback + checks + findings + footer
    start = base_page.index('<main class="log-content"')
    end = base_page.index('</main>', start) + len('</main>')
    page = base_page[:start] + '<main class="log-content" id="log-2026-10-29">' + body + '</main>' + base_page[end:]
    page = page.replace('href="#log-2026-10-25"', 'href="#log-2026-10-29"')
    page = page.replace('<span>10.25</span>', '<span>10.29</span>')
    page = page.replace('<title>cecilia-research-log · 2026.10.25</title>', '<title>cecilia-research-log · 2026.10.29 · Edge × Nano 推理</title>')
    old_description = 'Cecilia 的研究日志：模型设计与训练细节、定性与定量结果、Findings。Stage 3 蒸馏，L40S / A5000 优化后推理结果。'
    page = page.replace(old_description, '已训练 Edge / Nano 四步学生的混合推理：同步视频、阶段质量、FPS 与历史反馈实验。全量 FVD 评测中。')
    return with_date_navigation(page, '2026-10-29')
