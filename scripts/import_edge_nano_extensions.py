"""Import two completed 172-clip route audits; no polling or Git operations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
DATA = REPO/'data/2026-10-29/edge-nano-4444'
ROUTES = {'nano_first_last': 'NEEN', 'nano_first_two': 'NNEE'}


def read(path):
    return json.loads(path.read_text())


def put(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def import_extensions(neen, nnee):
    expected_protocol = read(DATA/'protocol.json')['fps_protocol']
    output = {'matched_count': 172, 'provenance_verified': True,
              'observed_at': datetime.now(timezone.utc).isoformat(),
              'variants': {}, 'benchmarks': {}, 'source_audit_sha256': {}, 'job_ids': []}
    per_clip = []
    for name, root in [('nano_first_last', neen), ('nano_first_two', nnee)]:
        path = root/'validated-extension.json'
        audit = read(path)
        assert audit['matched_count'] == 172
        assert audit['matched_indices'] == [f'{i:04d}' for i in range(172)]
        for key in ('all_noise_actions_matched', 'reference_features_exactly_reused',
                    'native_parity_passed', 'fvd_independently_crosschecked'):
            assert audit[key], f'failed provenance check: {key}'
        assert audit['placement'] == 'three-a5000' and audit['gpu_count'] == 3 and 'A5000' in audit['gpu']
        assert set(audit['variants']) == {name}
        q, bench = audit['variants'][name], audit['benchmark']
        assert q['count'] == 172 and q['route'] == ROUTES[name]
        assert bench['protocol'] == expected_protocol and bench['gpu_count'] == 3
        assert bench['warmups_per_variant'] == 1 and bench['timed_repeats'] == 3 and bench['index'] == 57
        assert set(bench['variants']) == {name}
        measured = bench['variants'][name]
        times = [r['combined_s'] for r in measured['repeats']]
        assert len(times) == 3 and all(math.isfinite(t) and t > 0 for t in times)
        assert math.isclose(q['warmed_fps'], 120/statistics.median(times), rel_tol=1e-10)
        for kind in ('full121', 'uniform16'):
            assert math.isfinite(q['fvd_'+kind])
            assert abs(q['fvd_'+kind]-q['fvd_'+kind+'_covariance_check']) < .01
        assert [r['index'] for r in audit['per_clip_quality']] == list(range(172))
        assert all(r['variant'] == name for r in audit['per_clip_quality'])
        output['variants'][name] = q
        output['benchmarks'][name] = bench
        output['job_ids'].append(audit['job_id'])
        output['source_audit_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        per_clip += audit['per_clip_quality']
    # These objects contain only aggregate metrics, timing and per-clip quality.
    # The private I3D vectors remain in the evaluation output directories.
    put(DATA/'route-extensions.json', output)
    put(DATA/'route-extensions-per-clip-quality.json', per_clip)
    existing = read(DATA/'validated-summary.json')
    existing['variants'].update(output['variants'])
    existing['completed_per_variant'].update({name: 172 for name in ROUTES})
    existing['extension_job_ids'] = output['job_ids']
    put(REPO/'docs/edge-nano-results.json', existing)
    protocol = read(DATA/'protocol.json')
    protocol.update(observed_at=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                    updated_on=datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat(),
                    extension_jobs=output['job_ids'])
    protocol.pop('access_note', None)
    put(DATA/'protocol.json', protocol)
    print(json.dumps({'imported_routes': list(ROUTES.values()), 'matched_count': 172,
                      'git_push_performed': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neen-root', type=Path, required=True)
    parser.add_argument('--nnee-root', type=Path, required=True)
    args = parser.parse_args()
    import_extensions(args.neen_root, args.nnee_root)
