"""One-shot audit of completed NEEN/NNEE validation, without publishing features."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

import numpy as np

from analyze_results import fvd_low_rank
from summarize import fvd

ROUTES = {'nano_first_last': 'NEEN', 'nano_first_two': 'NNEE'}
INDICES = [f'{i:04d}' for i in range(172)]


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate(root, baseline):
    root, baseline = Path(root), Path(baseline)
    settings, original = read(root/'settings.json'), read(baseline/'settings.json')
    names = list(settings['variants'])
    require(names and set(names) <= set(ROUTES), 'expected NEEN/NNEE extension variants')
    for name in names:
        require(settings['variants'][name] == {'route': ROUTES[name]}, 'unexpected history/diagnostic policy')
    for key in ('sampling', 'prompt', 'decoder', 'fps_protocol', 'torch'):
        require(settings[key] == original[key], f'baseline protocol mismatch: {key}')
    for model in ('E', 'N'):
        for key in ('payload_sha256', 'params', 'base_revision'):
            require(settings['models'][model].get(key) == original['models'][model].get(key),
                    f'baseline model mismatch: {model}/{key}')
    require(settings['indices'] == list(range(172)), 'not the full validation index')
    require(settings['placement'] == 'three-a5000' and settings['gpu_count'] == 3
            and 'A5000' in settings['gpu'], 'FPS requires the matched three-A5000 deployment')
    parity = read(root/'parity.json')
    require(all(parity[k]['exact'] and parity[k]['max_abs'] == 0 for k in ('E', 'N')),
            'native pure-model parity failed')
    require(parity['placement']['exact'] and parity['placement']['generated_chunks'] == 2,
            'layer placement parity failed')
    require(parity['feedback']['shadow_final_exact'] and parity['feedback']['first_chunk_exact'],
            'feedback isolation failed')
    bench = read(root/'benchmark.json')
    require(bench['placement'] == settings['placement'] and bench['gpu_count'] == 3
            and bench['gpu'] == settings['gpu'], 'benchmark hardware differs from settings')
    require(bench['index'] == 57 and bench['warmups_per_variant'] == 1
            and bench['timed_repeats'] == 3, 'benchmark protocol mismatch')
    require(set(bench['variants']) == set(names) and bench['protocol'] == settings['fps_protocol'],
            'benchmark variants/timing scope mismatch')
    results, per_clip = {}, []
    real_features = {kind: [] for kind in ('full121', 'uniform16')}
    for index in INDICES:
        ref = read(root/'reference'/index/'record.json')
        require(ref == read(baseline/'reference'/index/'record.json'), f'reference mismatch: {index}')
        with np.load(root/'reference'/index/'features.npz') as cached, \
                np.load(baseline/'reference'/index/'features.npz') as source:
            for kind in real_features:
                require(np.array_equal(cached[kind], source[kind]), f'I3D reference mismatch: {index}/{kind}')
                real_features[kind].append(cached[kind])
    for name in names:
        found = sorted(p.parent.name for p in (root/name).glob('*/metrics.json'))
        require(found == INDICES, f'{name}: full172 required, found {len(found)}')
        rows, features = [], {kind: [] for kind in real_features}
        route = ROUTES[name]
        for index in INDICES:
            row = read(root/name/index/'metrics.json')
            source = read(baseline/'edge4444'/index/'metrics.json')
            for key in ('index', 'cid', 'seed', 'initial_noise_sha256', 'actions_sha256'):
                require(row[key] == source[key], f'input mismatch: {name}/{index}/{key}')
            require(row['variant'] == name and not row['diagnostic_not_deployment_speed'], 'wrong variant')
            gen = row['generation']
            require(gen['route'] == route and not gen['diagnostic'] and gen['edge_history'] == 'final',
                    f'history policy mismatch: {name}/{index}')
            grid = gen['grid']
            require(grid == [1., .9375, .8333333134651184, .625, 0.], 'time grid mismatch')
            require(gen['counts'] == {k: {'denoise': 10*route.count(k), 'shadow': 0, 'commit': 11}
                                      for k in ('E', 'N')}, 'incorrect model call counts')
            expected_calls = [('E', 'commit', 0, 0., 0.), ('N', 'commit', 0, 0., 0.)]
            for block in range(1, 11):
                expected_calls += [(k, 'denoise', block, grid[i], grid[i+1]) for i, k in enumerate(route)]
                expected_calls += [(k, 'commit', block, 0., 0.) for k in ('E', 'N')]
            calls = [(c['model'], c['kind'], c['block'], c['source'], c['target']) for c in gen['calls']]
            require(calls == expected_calls, f'wrong denoising/commit order: {name}/{index}')
            q = row['quality']
            require(len(q['lpips_alex_per_block']) == 10 and np.isfinite(
                [q['psnr_db'], q['ssim'], q['lpips_alex'], *q['lpips_alex_per_block']]).all(), 'nonfinite quality')
            timing = row['timing']
            require(timing['combined_s'] > 0 and math.isclose(
                timing['combined_s'], timing['rollout_s']+timing['decode_s'], rel_tol=1e-10), 'invalid timing')
            with np.load(root/name/index/'features.npz') as values:
                for kind in features:
                    require(values[kind].shape == (400,) and np.isfinite(values[kind]).all(), 'invalid I3D features')
                    features[kind].append(values[kind])
            rows.append(row)
            per_clip.append({'index': int(index), 'variant': name, 'quality': q})
        result = {key: float(np.mean([r['quality'][key] for r in rows]))
                  for key in ('lpips_alex', 'psnr_db', 'ssim')}
        result.update(route=route, count=172)
        for kind in features:
            a, b = np.stack(features[kind]), np.stack(real_features[kind])
            value, check = fvd_low_rank(a, b), fvd(a, b)
            require(math.isfinite(value) and abs(value-check) < .01, 'independent FVD implementations disagree')
            result['fvd_'+kind] = value
            result['fvd_'+kind+'_covariance_check'] = check
        measured = bench['variants'][name]
        times = [r['combined_s'] for r in measured['repeats']]
        require(len(times) == 3 and all(math.isfinite(t) and t > 0 for t in times), 'invalid benchmark samples')
        require(not measured['diagnostic'] and math.isclose(measured['fps_from_median_seconds'],
                120/statistics.median(times), rel_tol=1e-10), 'FPS inconsistent with repeated timing')
        result['warmed_fps'] = measured['fps_from_median_seconds']
        results[name] = result
    # No raw I3D vectors or full model settings in this publication-ready object.
    report = {'matched_count': 172, 'matched_indices': INDICES, 'all_noise_actions_matched': True,
              'reference_features_exactly_reused': True, 'native_parity_passed': True,
              'fvd_independently_crosschecked': True, 'variants': results, 'benchmark': bench,
              'job_id': settings['job_id'], 'gpu': settings['gpu'], 'gpu_count': 3,
              'placement': settings['placement'], 'observed_at': datetime.now(timezone.utc).isoformat(),
              'per_clip_quality': per_clip}
    path = root/'validated-extension.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.root, args.baseline)
    print(json.dumps({'matched_count': report['matched_count'], 'variants': report['variants']}, indent=2))
