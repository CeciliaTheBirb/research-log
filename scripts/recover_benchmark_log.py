"""Recover the benchmark's printed measurements and verify its printed summary.

These two jobs wrote benchmark.json under temporary job code, which cleanup removed.
Their persistent Slurm stdout retained all 27 timed records and the final summary.
Runtime metadata below comes from the audited benchmark source, not printed fields.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument('hardware', choices=['l40s', 'a5000'])
p.add_argument('log', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
raw = a.log.read_bytes()
log = raw.decode()
job = re.search(r'job (\d+) \(research-' + a.hardware + r'\) gpus=1', log).group(1)
assert 'allocated GPUs are idle' in log
assert 'phase run done' in log, 'Wait for successful job completion'
objects = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
records = [x for x in objects if 'variant' in x and 'repeat' in x]
ends = [x for x in objects if x.get('complete')]
assert len(records) == 27 and len(ends) == 1
summary = ends[0]['summary']
assert len(summary) == 3
for key, row in summary.items():
    assert key == row['variant'] + '__' + row['runtime']
    selected = [x for x in records if x['variant'] == row['variant'] and x['runtime'] == row['runtime']]
    assert {(x['index'], x['repeat']) for x in selected} == {(i, j) for i in [0, 57, 114] for j in range(3)}
    for i in [0, 57, 114]:
        for field in ['rollout_s', 'decode_s', 'combined_s']:
            expected = median(x[field] for x in selected if x['index'] == i)
            assert math.isclose(row['per_clip_median'][str(i)][field], expected, rel_tol=1e-12)
    seconds = sum(row['per_clip_median'][str(i)]['combined_s'] for i in [0, 57, 114])
    assert math.isclose(row['combined_fps'], 360 / seconds, rel_tol=1e-12)
checks = [x['text_cache'] for x in records if 'text_cache' in x]
assert len(checks) == 6 and all(x['enabled'] and x['layer_count'] == 28 for x in checks)
result = {
    'complete': True,
    'gpu': {'l40s': 'NVIDIA L40S', 'a5000': 'NVIDIA RTX A5000'}[a.hardware],
    'torch': re.search(r'torch: (\S+)', log).group(1),
    'job_id': job, 'indices': [0, 57, 114], 'warmup_per_clip': 1, 'timed_repeats_per_clip': 3,
    'protocol': '120 new RGB frames / synchronized generation plus TinyVAE BF16 streaming decode; '
                'same checkpoints, prompts, action layout and canonical noise on each GPU. '
                'Every video rebuilds its own text cache; loading, anchor encoding, CPU conversion, '
                'MP4 and quality validation excluded.',
    'records': records, 'summary': summary,
    'runtimes': {
        'original_native': {'single_device': True, 'fsdp': False, 'text_cache': True,
                            'source': 'Audited source asserts cache_text and absence of FSDP before timed records.'},
        'single_device_text_cache': {'single_device': True, 'fsdp': False, 'text_cache': True,
                                     'source': 'Selected optimized-only runtime and six printed cache checks.'}
    },
    'provenance': {
        'source': 'Recovered from persistent Slurm stdout JSON, including the successful final summary.',
        'reason': 'Relative benchmark.json output was inside temporary job code and removed during normal cleanup.',
        'log_sha256': hashlib.sha256(raw).hexdigest(),
        'metadata': 'GPU is the audited scheduler node hardware; runtime flags and warmup are from the submitted source.',
        'validation': 'All 27 timed records retained; independently recomputed medians and aggregate FPS match printed summary.'
    }
}
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(a.hardware, 'recovered and verified 27 timed records from job', job)
