"""Import a completed 172-episode hybrid evaluation and optional hardware timings."""
from pathlib import Path
import json
import shutil

repo = Path(__file__).resolve().parents[1]
source = repo/'data/2026-10-25/hf-hybrid'
summary = json.loads((source/'summary.json').read_text())
assert summary['complete'] and summary['count'] == 172
name = 'hybrid2400_nocaption'
row = summary['variants'][name]
results = json.loads((repo/'docs/results.json').read_text())
results['models'] = [m for m in results['models'] if m['id'] != 'hybrid']
entry = {'id':'hybrid','name':'43222+hybrid forcing','checkpoint':'HF Edge · step2400',
         'fvd121':row['fvd']['full121'],'fvd16':row['fvd']['uniform16'],
         'lpips':row['quality']['lpips_alex']['mean'],'psnr':row['quality']['psnr_db']['mean'],
         'ssim':row['quality']['ssim']['mean'],'calls':34}
ci = source/'fvd-uncertainty.json'
if ci.exists():
    entry['fvd_delta'] = json.loads(ci.read_text())[name+'_minus_old4444_native']
results['models'].append(entry)
results['hf_source'] = json.loads((source/'source.json').read_text())
results['quality_gpu']['hybrid'] = 'NVIDIA RTX A5000'
results['hybrid_evaluation'] = {'episodes':172,'checkpoint_step':2400,'runtime_commit':'dec58fab57464baab30f664ee78769e5dc5fae54',
                              'schedule':[4,3,2,2,2,2,2,2,2,2],'window':{'sink_size':1,'history_cutoff':2},
                              'history_memory':True,'prompt':'','action_layout':'120 transitions',
                              'external_training_overlap':'not established'}
for gpu in ['a5000','l40s']:
    f = source/f'{gpu}-benchmark.json'
    if f.exists():
        benchmark = json.loads(f.read_text())
        assert benchmark['complete'] and len(benchmark['records']) == 9
        assert benchmark['indices'] == [0,57,114]
        medians = benchmark['summary']['per_clip_median']
        fps = 360/sum(r['combined_s'] for r in medians.values())
        assert abs(fps-benchmark['summary']['combined_fps']) < 1e-10
        results['hardware_benchmarks'][gpu]['models']['hybrid'] = {'fps':fps,'per_clip_median':medians,
                                                                 'job_id':int(benchmark['settings']['job_id'])}
    else:
        results['hardware_benchmarks'][gpu]['models']['hybrid'] = {'fps':None,'status':'queued','job_id':82729}
(repo/'docs/results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
metrics = [('fvd121',2,False),('fvd16',2,False),('lpips',4,False),('psnr',2,True),('ssim',4,True)]
best = {k:(max if higher else min)(m[k] for m in results['models']) for k,_,higher in metrics}
rows = []
for m in results['models']:
    cells = ''.join(f'<td'+(' class="best"' if m[k]==best[k] else '')+f'>{m[k]:.{precision}f}</td>' for k,precision,_ in metrics)
    rows.append(f'<tr class="{m["id"]}"><th scope="row"><span class="swatch"></span>{m["name"]}<small>{m["checkpoint"]}</small></th>{cells}</tr>')
table = ('<div class="table-wrap"><table class="results-table"><caption>172 条验证视频的质量结果</caption><thead><tr>'
         '<th scope="col">模型</th><th scope="col">FVD121 ↓</th><th scope="col">FVD16 ↓</th><th scope="col">LPIPS ↓</th>'
         '<th scope="col">PSNR ↑ dB</th><th scope="col">SSIM ↑</th></tr></thead><tbody id="results-body">'+''.join(rows)+'</tbody></table></div>')
(repo/'scripts/quality-table.html').write_text(table+'\n')
print(json.dumps(entry,ensure_ascii=False,indent=2))
