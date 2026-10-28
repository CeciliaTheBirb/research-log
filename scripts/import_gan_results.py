"""Audit and import the completed conditional video GAN ablation."""
import argparse
import hashlib
import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument('--input-root', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
dest = repo / 'data/2026-10-26/noisy-conditional-gan'
source = args.input_root
read = lambda name: json.loads((source / name).read_text())
write = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
summary, benchmark, uncertainty = [read(n) for n in ['summary.json', 'benchmark-a5000/benchmark.json', 'fvd-uncertainty.json']]
assert summary['complete'] and summary['count'] == 172
assert benchmark['complete'] and benchmark['indices'] == [0, 57, 114]
assert benchmark['warmup_per_clip'] == 1 and benchmark['timed_repeats_per_clip'] == 3
assert len(benchmark['records']) == 27 and int(benchmark['job_id']) == 83084
assert all(v['single_device'] and v['text_cache'] for v in benchmark['runtimes'].values())
baseline, candidate, old = 'noisy_matched1600_nocaption', 'noisy_gan1600_nocaption', 'old4444_native'
manifest = read('manifest.json')
base_spec, gan_spec = [v for v in manifest['variants'] if v['name'] in [baseline, candidate]]
assert base_spec['policy'] == gan_spec['policy'] and base_spec['conditioning'] == gan_spec['conditioning']
assert manifest['experiment']['student_updates'] == manifest['experiment']['fake_updates'] == 800
episodes = []
for i in range(172):
    records = {name: read(f'{name}/{i:04d}/metrics.json') for name in [old, baseline, candidate]}
    b, g = records[baseline], records[candidate]
    assert all(row['index'] == i and row['cid'] == b['cid'] and row['seed'] == manifest['seed'] + i for row in records.values())
    assert len({row['timing']['initial_noise_sha256'] for row in records.values()}) == 1
    assert b['prompt_tokens'] == g['prompt_tokens'] and b['prompt_ids_sha256'] == g['prompt_ids_sha256']
    assert b['timing']['calls'] == g['timing']['calls'] and g['timing']['calls']['model_calls'] == 24
    episodes.append({'index': i, 'cid': b['cid'], 'seed': b['seed'],
                     'initial_noise_sha256': b['timing']['initial_noise_sha256'],
                     'variants': {n: {k: row[k] for k in ['prompt_tokens', 'prompt_ids_sha256', 'quality', 'timing']} for n, row in records.items()}})
for name in [old, baseline, candidate]:
    for metric in ['psnr_db', 'ssim', 'lpips_alex']:
        mean = statistics.mean(row['variants'][name]['quality'][metric] for row in episodes)
        assert math.isclose(mean, summary['variants'][name]['quality'][metric]['mean'], abs_tol=1e-12)
for name, entry in benchmark['summary'].items():
    records = [r for r in benchmark['records'] if r['variant'] == entry['variant'] and r['runtime'] == entry['runtime']]
    assert len(records) == 9
    medians = [statistics.median(r['combined_s'] for r in records if r['index'] == i) for i in benchmark['indices']]
    assert math.isclose(360 / sum(medians), entry['combined_fps'], abs_tol=1e-12)
comparison = candidate + '_minus_' + baseline
assert math.isclose(uncertainty[comparison]['observed_delta'], summary['paired_comparisons'][comparison]['fvd_delta']['full121'], abs_tol=1e-9)
write(dest / 'summary.json', summary)
write(dest / 'fvd-uncertainty.json', uncertainty)
write(dest / 'a5000-benchmark.json', {k: v for k, v in benchmark.items() if k != 'manifest'})
(dest / 'episode-metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in episodes))
write(dest / 'evaluation-audit.json', {'episodes': 172, 'same_cid_seed': True, 'same_initial_noise': True,
      'same_noisy_prompt_tokens_and_hash': True, 'same_noisy_policy_and_conditioning': True,
      'noisy_calls_24': True, 'paired_metrics_recomputed': True, 'speed_medians_recomputed': True,
      'reference_feature_consistency': summary['reference_consistency'],
      'raw_benchmark_sha256': hashlib.sha256((source / 'benchmark-a5000/benchmark.json').read_bytes()).hexdigest()})
rows = [json.loads(line) for line in (source / 'training/metrics.jsonl').read_text().splitlines()]
run = read('training/run.json')
meta = read('training/checkpoint-meta.json')
assert run['status'] == 'finished' and run['step'] == meta['step'] == 1600
assert meta['video_gan']['weight'] == 0.001
assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v, (int, float)))
student = [r for r in rows if 'train/student_loss' in r]
fake = [r for r in rows if 'train/fake_loss' in r]
assert len(student) == len(fake) == 800
(dest / 'training-metrics.jsonl').write_text((source / 'training/metrics.jsonl').read_text())
now = datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')
progress = {'observed_at': now, 'job': 83083, 'status': 'finished', 'step': 1600, 'total': 1600,
            'student_updates': 800, 'fake_and_discriminator_updates': 800, 'all_logged_values_finite': True,
            'latest_checkpoint': 1600, 'latest_checkpoint_box_verified': True, 'gan_optimizer_box_verified': True,
            'peak_allocated_gib': max(r.get('mem/max_alloc_gb', 0) for r in rows),
            'validation': [{k: v for k, v in row.items() if k == 'step' or k.startswith('val/')} for row in rows if 'val/loss' in row],
            'gan_endpoint_gradient_ratios': [{k: v for k, v in row.items() if k in ['step', 'train/gan_weight', 'train/gan_to_dmd_endpoint_grad_ratio']} for row in rows if 'train/gan_to_dmd_endpoint_grad_ratio' in row],
            'r1_nonzero_updates': sum(r.get('train/gan_r1', 0) > 0 for r in rows)}
write(dest / 'progress.json', progress)
experiment = json.loads((dest / 'experiment.json').read_text())
experiment.update(status='trained_quality_complete_l40s_running', status_zh='800次student＋800次fake/GAN更新已完成；172条质量及A5000测速完成，L40S测速运行中',
                  updated_at=now, progress=progress, quality=summary['variants'][candidate],
                  paired_comparison=summary['paired_comparisons'][comparison], fvd_uncertainty=uncertainty[comparison],
                  training_elapsed_seconds=14 * 3600 + 6 * 60 + 28, training_gpu_hours=(14 * 3600 + 6 * 60 + 28) * 4 / 3600)
experiment['fps'] = {'a5000': {n: benchmark['summary'][n + '__' + ('original_native' if n == old else 'single_device_text_cache')]['combined_fps'] for n in [old, baseline, candidate]}, 'l40s': None}
l40s = None
if (source / 'benchmark-l40s/benchmark.json').exists():
    l40s = read('benchmark-l40s/benchmark.json')
    assert l40s['complete'] and l40s['indices'] == [0, 57, 114] and int(l40s['job_id']) == 83085
    assert len(l40s['records']) == 27 and l40s['warmup_per_clip'] == 1 and l40s['timed_repeats_per_clip'] == 3
    assert all(v['single_device'] and v['text_cache'] for v in l40s['runtimes'].values())
    for entry in l40s['summary'].values():
        records = [r for r in l40s['records'] if r['variant'] == entry['variant'] and r['runtime'] == entry['runtime']]
        medians = [statistics.median(r['combined_s'] for r in records if r['index'] == i) for i in l40s['indices']]
        assert len(records) == 9 and math.isclose(360 / sum(medians), entry['combined_fps'], abs_tol=1e-12)
    experiment['fps']['l40s'] = {n: l40s['summary'][n + '__' + ('original_native' if n == old else 'single_device_text_cache')]['combined_fps'] for n in [old, baseline, candidate]}
    experiment.update(status='completed', status_zh='训练、172条质量与A5000 / L40S重复测速均已完成')
    write(dest / 'l40s-benchmark.json', {k: v for k, v in l40s.items() if k != 'manifest'})
write(dest / 'experiment.json', experiment)
results_path = repo / 'docs/results.json'
results = json.loads(results_path.read_text())
q = summary['variants'][candidate]
model = {'id': 'noisy_gan', 'name': '4222+noisy kv reuse + video GAN', 'checkpoint': '1600次总更新 · 800次student', 'calls': 24,
         'fvd121': q['fvd']['full121'], 'fvd16': q['fvd']['uniform16'], 'lpips': q['quality']['lpips_alex']['mean'],
         'psnr': q['quality']['psnr_db']['mean'], 'ssim': q['quality']['ssim']['mean'], 'fvd_delta_vs_noisy': uncertainty[comparison]}
results['models'] = [m for m in results['models'] if m['id'] != 'noisy_gan'] + [model]
results['hardware_benchmarks']['a5000']['models']['noisy_gan'] = {
    'fps': experiment['fps']['a5000'][candidate], 'job_id': 83084,
    'per_clip_median': benchmark['summary'][candidate + '__single_device_text_cache']['per_clip_median']}
results['hardware_benchmarks']['l40s']['models']['noisy_gan'] = {'fps': experiment['fps']['l40s'][candidate] if l40s else None, 'job_id': 83085, 'status': 'completed' if l40s else 'running'}
results['gan_evaluation'] = {'job_id': 83084, 'quality_gpu': benchmark['gpu'], 'episodes': 172,
                             'paired_comparison': experiment['paired_comparison'], 'same_job_speed': experiment['fps']['a5000']}
results['updated'] = now
write(results_path, results)
fields = [('fvd121', '.2f', False), ('fvd16', '.2f', False), ('lpips', '.4f', False), ('psnr', '.2f', True), ('ssim', '.4f', True)]
best = {k: (max if high else min)(m[k] for m in results['models'] if m['id'] != 'noisy_gan') for k, _, high in fields}
table_rows = []
for model in results['models']:
    if model['id'] == 'noisy_gan':
        continue
    cells = ''.join(f'<td{" class=\"best\"" if model[k] == best[k] else ""}>{model[k]:{spec}}</td>' for k, spec, _ in fields)
    table_rows.append(f'<tr class="{model["id"].replace("_", "-")}"><th scope="row">{html.escape(model["name"])}<small>{html.escape(model["checkpoint"])}</small></th>{cells}</tr>')
header = '<div class="table-wrap"><table class="results-table"><caption>172 条验证视频的质量结果</caption><thead><tr><th scope="col">模型</th><th scope="col">FVD121 ↓</th><th scope="col">FVD16 ↓</th><th scope="col">LPIPS ↓</th><th scope="col">PSNR ↑ dB</th><th scope="col">SSIM ↑</th></tr></thead><tbody id="results-body">'
(repo / 'scripts/quality-table.html').write_text(header + ''.join(table_rows) + '</tbody></table></div>\n')
print('Imported 172 audited episodes, 27 verified timing records and completed 800:800 training.')
