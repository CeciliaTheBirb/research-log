"""Import a completed CoStream benchmark without publishing machine-local paths."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('hardware', choices=['l40s', 'a5000'])
p.add_argument('benchmark', type=Path)
a = p.parse_args()
r = Path(__file__).resolve().parents[1]
b = json.loads(a.benchmark.read_text())
assert b['complete'] and b['indices'] == [0, 57, 114]
assert b['timed_repeats_per_clip'] == 3 and b['warmup_per_clip'] == 1
assert b['runtimes']['original_native']['single_device']
assert b['runtimes']['original_native']['text_cache']
assert b['runtimes']['single_device_text_cache']['single_device']
assert b['runtimes']['single_device_text_cache']['text_cache']
assert len(b['records']) == 27
mapping = [('old', 'old4444_native', 'original_native'),
           ('clean', 'clean3200_nocaption', 'single_device_text_cache'),
           ('noisy', 'noisy_matched1600_nocaption', 'single_device_text_cache')]
summary = {label: b['summary'][name + '__' + mode] for label, name, mode in mapping}
public = {k: b[k] for k in ['gpu', 'torch', 'job_id', 'indices', 'warmup_per_clip',
                            'timed_repeats_per_clip', 'protocol', 'records', 'runtimes', 'complete']}
public['summary'] = summary
public['provenance'] = b.get('provenance', {})
public['model_labels'] = {name: label for label, name, _ in mapping}
(r / f'data/2026-10-25/{a.hardware}-benchmark.json').write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
f = r / 'docs/results.json'
s = json.loads(f.read_text())
s['hardware_benchmarks'][a.hardware] = {'status': 'completed', 'gpu': b['gpu'], 'job_id': int(b['job_id']),
    'indices': b['indices'], 'warmup_per_clip': 1, 'timed_repeats_per_clip': 3,
    'models': {label: {'fps': row['combined_fps'], 'per_clip_median': row['per_clip_median']} for label, row in summary.items()}}
s['old_native_runtime'] = b['runtimes']['original_native']
f.write_text(json.dumps(s, ensure_ascii=False, indent=2) + '\n')
print(a.hardware, {label: round(row['combined_fps'], 3) for label, row in summary.items()})
