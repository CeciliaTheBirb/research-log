"""Render the completed native-model pilot from archived records and videos."""
from pathlib import Path
import html
import json
import re

from report_dates import with_date_navigation


def build_native_follow_page(base_page, root):
    data = root / 'data/2026-10-29/native-forward-encounter'
    s = json.loads((data / 'summary.json').read_text())
    assert s['status'] == 'completed' and s['total_clips'] == 36
    asset = 'assets/native-forward-encounter'
    github = 'https://github.com/CeciliaTheBirb/research-log/tree/main/data/2026-10-29/native-forward-encounter'
    videos = []
    for scene, title in [('rain', '雨后路口'), ('night0057', '夜间人行道')]:
        for seed in [2026, 2027]:
            stem = f'{scene}-seed{seed}'
            bundle = next(b for b in s['bundles'] if b['file'] == f'{stem}-joint-fd.mp4')
            notes = {
                ('rain', 2026): '三种模型背景视点几乎不前进。左侧事件中行人较远且不稳定；右侧事件常变成从画面右边进入、随后横向或背向走远。',
                ('rain', 2027): '仍未看到稳定的机器人前进。Edge / Nano 的右侧事件主要变成横穿，Super 的行人生成较弱。两侧均未稳定完成指定擦身经过。',
                ('night0057', 2026): 'Super 有明显的镜头前进，但近景物体出现形变；Edge / Nano 的镜头位移较弱。三者左右事件均未稳定完成“迎面靠近后从指定一侧经过”。',
                ('night0057', 2027): 'Super 没有复现 seed 2026 的明显前进。行人事件有变化，但多数停留在远处、横穿或走远，左右擦身要求仍未满足。',
            }[(scene, seed)]
            links = ' · '.join(
                f'<a href="{asset}/{stem}-{case}.mp4" target="_blank" rel="noopener">{label} ↗</a>'
                for case, label in [('fd_forward_control', '只前进'), ('fd_forward_oncoming_left', '左侧经过'), ('fd_forward_oncoming_right', '右侧经过')])
            videos.append(f'''<article class="rain-results" id="{stem}">
<h3 class="result-title">{title} · seed {seed}</h3>
<div class="viewer-head"><p>左 Edge / 中 Nano / 右 Super<br>依次：只前进 → 迎面行人从机器人左侧经过 → 右侧经过</p><a href="{asset}/{bundle['file']}" download>下载合辑 ↓</a></div>
<div class="video-shell"><video controls playsinline preload="metadata" poster="{asset}/{bundle['poster']}" aria-label="{title} seed {seed} 原生三模型同步对比"><source src="{asset}/{bundle['file']}" type="video/mp4">请下载 MP4 观看。</video></div>
<p class="caption-note">每段6.05秒，合辑18.15秒；标题表示输入要求，实际完成情况见下文。单独观看：{links}。</p>
<p><strong>观看观察：</strong>{notes}</p></article>''')
    runtime_rows = ''.join(
        f'<tr><th scope="row">Cosmos3-{m["name"]}</th><td>{m["gpu_count"]}×{html.escape(m["gpu"].removeprefix("NVIDIA "))}</td><td>{m["median_total_seconds"]:.2f} s</td><td>{m["median_fps"]:.3f}</td></tr>'
        for m in s['models'])
    model_rows = ''.join(
        f'<tr><th scope="row">Cosmos3-{m["name"]}</th><td><a href="https://huggingface.co/{m["checkpoint"]["repo_id"]}/tree/{m["checkpoint"]["revision"]}" target="_blank" rel="noopener">官方权重 ↗</a><small>revision {m["checkpoint"]["revision"][:12]}</small></td><td>{m["job_id"]} · 12 / 12 完成</td></tr>'
        for m in s['models'])
    body = f'''<header class="entry-header"><div class="page-icon" aria-hidden="true">▤</div><div class="entry-date">2026 年 10 月 29 日</div>
<h1>原生 Super / Nano / Edge</h1><p class="entry-deck">机器人向前走，行人迎面靠近并从左／右侧经过。36条原生模型视频已完成；本轮没有显示稳定的动作与事件联合跟随，暂不足以据此决定 Super 的第一阶段训练。</p>
<div class="entry-tags"><span>原生权重 · 无 LoRA</span><span>2场景 × 2 seeds × 3条件</span><span>前进 0.5 m/s</span><span>TinyVAE BF16 流式</span></div></header>
<section id="results" class="log-section"><h2>前进＋迎面行人 / 全部对比视频</h2>
<p>三种模型收到相同的首帧、前进数值动作、初始随机噪声和事件文本。每种模型12条，共36条；每个场景保留两个固定seed的全部结果。左右以机器人摄像头画面为准。</p>
<div class="timing-box"><h4>输入要求不等于模型已经做到。</h4><p>输入为0.5 m/s直行、零转向，6秒共请求前进3米。行人从约6米前方迎面靠近，再从指定一侧擦身并离开近端画面。3米是输入轨迹，未从生成视频测出机器人实际位移。</p></div>
{''.join(videos)}</section>
<section id="findings" class="log-section findings-section"><h2>观察与局限</h2><ol class="finding-list">
<li><h3>Super 没有稳定胜出。</h3><p>夜景seed 2026中，Super的前进视差更明显，但另一seed没有复现，雨后路口中三者均近似静止。不能把单个片段当成整体优势。</p></li>
<li><h3>左右文本会改变视频，但完整事件经常失败。</h3><p>行人经常变成远处活动、横穿或背向走远。出现行人、从某一侧进入，均不等于完成“迎面靠近后从指定一侧经过”。这里保留人工观察，不虚构自动成功率。</p></li>
<li><h3>当前结果还不能单独归因于模型大小或机器人域未微调。</h3><p>使用原生基础权重，在CoStream移植推理后端运行，未加载机器人数据的适配器。输入格式检查通过，完整权重的官方运行时数值对齐仍未完成；需要再用官方运行时及已知有效动作样例作对照，区分域差异与推理差异。</p></li>
<li><h3>这一轮同时测试文本与动作的联合要求。</h3><p>文本和数值动作都要求前进，所以成片即使前进，也不能单独证明模型使用了数值动作。独立动作跟随仍需要固定文本、改变动作的完整对照。</p></li></ol>
<p class="caption-note">两场景、两seed属于探索性诊断。此轮未估计总体FVD；FVD本身无需逐视频配对，但需要合适的真实／生成样本集合。反事实行人与动作没有配对真实未来，不能报告PSNR／SSIM。10.25与10.27的172条validation指标属于不同实验，不移作本轮原生模型指标。</p></section>
<section id="runtime" class="log-section"><h2>本轮实际推理耗时</h2><div class="table-wrap"><table><thead><tr><th scope="col">模型</th><th scope="col">实际分配GPU</th><th scope="col">6秒视频耗时中位数</th><th scope="col">新帧 FPS</th></tr></thead><tbody>{runtime_rows}</tbody></table></div>
<p class="caption-note">12条不同条件各推理一次，表中为其中位数，未做重复测速。计时包含生成＋TinyVAE解码，排除模型加载、首帧编码、CPU转换与写文件；FPS以120个新帧计算。</p>
<p>三个模型的硬件分配不同，Edge用请求内文本缓存，Nano / Super用多卡序列并行和权重分片，当前未启用文本缓存。因此这些耗时记录用于说明本轮成本，不能作为同硬件速度排名。全部为50步原生双向扩散，不能与此前4222蒸馏学生的FPS直接比较。</p></section>
<section id="protocol" class="log-section"><h2>相同测试条件</h2><div class="common-settings"><dl>
<div><dt>输入与动作</dt><dd>雨后路口、night0057首帧；seed 2026 / 2027。120行AV9相对动作，每行平移[0,0,0.025]米、旋转为单位旋转；无anchor占位动作。</dd></div>
<div><dt>三种事件</dt><dd>无新增前景行人／迎面左侧经过／迎面右侧经过。三种事件的数值动作完全相同，事件文本不同。</dd></div>
<div><dt>生成设置</dt><dd>480×832，121帧，20 FPS；50步UniPC，forward dynamics，guidance=1，flow shift=5，action domain=av，原生action JSON文本。TinyVAE step38000 BF16流式解码。</dd></div>
<div><dt>首帧处理</dt><dd>首图重复至121帧后编码，仅latent第0帧作条件；未来真实视频不作为输入。三模型复用同一首帧latent与CPU FP32初始噪声。</dd></div>
<div><dt>已完成核查</dt><dd>36条输出完整；跨模型图像、anchor、初始噪声、动作、文本与manifest一致；文本无截断。三套完整权重键名／形状检查通过；8组缩小模型FP32移植对齐最大误差为0。</dd></div>
<div><dt>核查边界</dt><dd>缩小模型对齐不等于完整checkpoint的BF16运行时对齐。本轮使用CoStream后端，不能标作官方服务端完整复现；原生权重与蒸馏学生分开记录。</dd></div></dl></div></section>
<section id="provenance" class="log-section"><h2>模型与原始记录</h2><div class="table-wrap"><table><thead><tr><th scope="col">模型</th><th scope="col">固定官方版本</th><th scope="col">Slurm作业</th></tr></thead><tbody>{model_rows}</tbody></table></div>
<p class="caption-note">输入、逐条输出hash与计时、配置、拼接审核记录和评测脚本均归档至仓库。视频为压缩观看副本；保留两seed的失败样例。旧静止预览属于先前独立诊断，未混入本页。</p>
<p><a href="{github}" target="_blank" rel="noopener">实验记录与脚本 ↗</a> · <a href="data/native-forward-encounter.json" download>本轮摘要 JSON ↓</a> · <a href="https://huggingface.co/nvidia/Cosmos3-Super/blob/f543c56225b2e04d0ad141e29655be3a45d9c455/README.md" target="_blank" rel="noopener">官方Super模型卡 ↗</a></p></section>
<footer class="entry-footer"><a href="{github}" target="_blank" rel="noopener">数据与复现记录 ↗</a><span>2026.10.29 · Cecilia</span></footer>'''
    start = base_page.index('<main ')
    end = base_page.index('</main>') + len('</main>')
    page = base_page[:start] + '<main class="log-content" id="log-2026-10-29">' + body + '</main>' + base_page[end:]
    page = page.replace('href="#log-2026-10-25"', 'href="#log-2026-10-29"')
    page = page.replace('<span>10.25</span></div>', '<span>10.29</span></div>')
    page = re.sub(r'<title>.*?</title>', '<title>cecilia-research-log · 2026.10.29 · 原生 Super / Nano / Edge</title>', page)
    page = re.sub(r'<meta name="description" content="[^"]*">', '<meta name="description" content="原生Cosmos3 Super、Nano、Edge的机器人前进与迎面行人联合控制实验：36条视频、四组合辑、条件核查与实际推理耗时。">', page)
    return with_date_navigation(page, '2026-10-29')
