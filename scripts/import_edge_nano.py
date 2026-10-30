"""Import completed, audited snapshots once; never poll, submit jobs or push Git."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data/2026-10-29/edge-nano-4444'
NAMES = ('edge4444', 'nano4444', 'nano_first', 'nano_last',
         'nano_last_feedback_diagnostic', 'nano_last_draft_history')


def read(root, name):
    return json.loads((root / name).read_text())


def put(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def import_results(full_root=None, l40s_root=None, launch=None):
    protocol = read(DATA, 'protocol.json')
    additions = {}
    if full_root:
        summary = read(full_root, 'validated-summary.json')
        paired = read(full_root, 'paired-analysis.json')
        if summary['matched_indices'] != [f'{i:04d}' for i in range(172)] or summary['matched_count'] != 172:
            raise ValueError('full matched172 analysis required; run analyze_results.py first')
        assert summary['all_noise_actions_matched'] and summary['all_shadow_final_latents_equal']
        assert paired['count'] == 172 and set(summary['variants']) == set(NAMES)
        # Cross-check the independently calculated final covariance-root FVD.
        native_summary = read(full_root, 'summary.json')
        assert native_summary['matched_count'] == 172
        for name in NAMES:
            for kind in ('full121', 'uniform16'):
                a = summary['variants'][name]['fvd_' + kind]
                b = native_summary['variants'][name]['fvd_' + kind]
                if not math.isfinite(a) or abs(a-b) > .01:
                    raise ValueError(f'FVD cross-check failed: {name}/{kind}: {a} vs {b}')
        additions.update({'validated-summary.json': summary, 'paired-analysis.json': paired,
                          'per-clip-quality.json': read(full_root, 'per-clip-quality.json')})
        protocol.update(status='complete: matched172 with paired session-bootstrap analysis',
                        observed_at=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat())
    if l40s_root:
        if not launch:
            raise ValueError('provide the Slurm launch record to establish single-L40S provenance')
        record = json.loads(launch.read_text())
        settings = read(l40s_root, 'settings.json')
        bench = read(l40s_root, 'benchmark.json')
        parity = read(l40s_root, 'parity.json')
        if record['gpu_count'] != 1 or record['partition'] != 'l40s' or 'L40S' not in bench['gpu']:
            raise ValueError('not the intended single-L40S benchmark')
        if settings.get('gpu_count', 1) != 1:
            raise ValueError('settings indicate multiple GPUs')
        if not all(parity[k]['exact'] and parity[k]['max_abs'] == 0 for k in ('E', 'N')):
            raise ValueError('native GPU parity did not pass')
        assert parity['feedback']['shadow_final_exact'] and parity['feedback']['first_chunk_exact']
        # The original pinned single-device runner may omit gpu_count in benchmark.json.
        # Establish this from the Slurm launch + runtime settings, not a guessed FPS ratio.
        assert bench['timed_repeats'] == 3 and bench['warmups_per_variant'] == 1 and bench['index'] == 57
        assert set(bench['variants']) == set(NAMES)
        expected = read(DATA, 'model-payload-sha256.json')
        actual = {name: value['payload_sha256'] for name, value in settings['models'].items()}
        if actual != expected:
            raise ValueError('L40S checkpoint payloads differ from the A5000 evaluation')
        if settings['sampling'] != protocol['sampling'] or settings['prompt'] != protocol['prompt']:
            raise ValueError('L40S sampler or prompt differs from the quality evaluation')
        if bench['protocol'] != read(DATA, 'benchmark.json')['protocol']:
            raise ValueError('L40S FPS timing protocol differs')
        if settings['decoder']['dtype'] != 'BF16' or settings['decoder']['parallel']:
            raise ValueError('L40S TinyVAE precision or streaming mode differs')
        for name in NAMES:
            row = bench['variants'][name]
            times = [r['combined_s'] for r in row['repeats']]
            if len(times) != 3 or any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError('invalid benchmark timings')
            if abs(row['fps_from_median_seconds'] - 120 / statistics.median(times)) > 1e-6:
                raise ValueError('FPS does not match three-repeat synchronized timing')
        bench.update(gpu_count=1, job_id=record['job_id'], placement='single-device',
                     provenance_verified=True, provenance='Slurm launch, runtime settings, native parity and matched payload hashes')
        additions['benchmark-l40s.json'] = bench
        protocol['l40s_job'] = {'id': record['job_id'], 'status': 'completed and verified', 'gpu_count': 1}
    if not additions:
        raise ValueError('provide at least one completed results directory')
    # Prepare and validate everything before replacing publication inputs.
    for obj in additions.values():
        json.dumps(obj, allow_nan=False)
    for name, obj in additions.items():
        put(DATA / name, obj)
    protocol.pop('access_note', None)
    put(DATA / 'protocol.json', protocol)
    if full_root:
        put(REPO / 'docs/edge-nano-results.json', additions['validated-summary.json'])
        put(REPO / 'docs/edge-nano-paired-analysis.json', additions['paired-analysis.json'])
    print(json.dumps({'imported': list(additions), 'git_push_performed': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-root', type=Path)
    p.add_argument('--l40s-root', type=Path)
    p.add_argument('--l40s-launch', type=Path)
    a = p.parse_args()
    import_results(a.full_root, a.l40s_root, a.l40s_launch)
