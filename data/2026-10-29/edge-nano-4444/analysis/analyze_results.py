"""Validate matched clips and quantify paired routing/history effects.

This is a one-shot CPU analysis. It does not launch or monitor GPU jobs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize import NAMES, fvd


def fvd_low_rank(x, y):
    """Exact covariance Frechet formula, using sample-space nuclear norm."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xc, yc = x-x.mean(0), y-y.mean(0)
    # Nonzero singular values of A.T@B give trace sqrt(Cx**.5 Cy Cx**.5).
    cross = xc @ yc.T / np.sqrt((len(x)-1)*(len(y)-1))
    covariance_trace = (xc*xc).sum()/(len(x)-1) + (yc*yc).sum()/(len(y)-1)
    return max(0., float(((x.mean(0)-y.mean(0))**2).sum() + covariance_trace
                         - 2*np.linalg.svd(cross, compute_uv=False).sum()))


def session_key(cid):
    parts = cid.split('__')
    if len(parts) < 3:
        raise ValueError('cannot identify source session')
    return '__'.join(parts[:3])


def cluster_draws(cids, repeats, seed):
    groups = {}
    for i, cid in enumerate(cids):
        groups.setdefault(session_key(cid), []).append(i)
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    return [np.concatenate([values[j] for j in rng.integers(len(values), size=len(values))])
            for _ in range(repeats)], len(values)


def paired_delta(candidate, baseline, draws, lower_is_better=True):
    delta = np.asarray(candidate, np.float64)-np.asarray(baseline, np.float64)
    observed = np.mean(delta, axis=0)
    boots = np.stack([delta[index].mean(axis=0) for index in draws])
    low, high = np.quantile(boots, [.025, .975], axis=0)
    result = {'mean_candidate_minus_baseline':observed.tolist(),
              'ci95':[low.tolist(), high.tolist()], 'lower_is_better':lower_is_better}
    if delta.ndim == 1:
        result.update(wins=int(np.sum(delta < 0 if lower_is_better else delta > 0)),
                      ties=int(np.sum(delta == 0)), total=len(delta))
    return result


def analyze(root, rgb_repeats=5000, fvd_repeats=512, seed=20261029):
    root = Path(root)
    sets = {n:{p.parent.name for p in (root/n).glob('*/metrics.json')} for n in NAMES}
    ids = sorted(set.intersection(*(sets[n] for n in NAMES)))
    if ids != [f'{i:04d}' for i in range(172)]:
        raise ValueError(f'Full paired analysis requires the same indices0..171 in all arms; got{len(ids)}')
    rows = {n:[json.loads((root/n/i/'metrics.json').read_text()) for i in ids] for n in NAMES}
    refs = [json.loads((root/'reference'/i/'record.json').read_text()) for i in ids]
    for j, index in enumerate(ids):
        for name in NAMES:
            row = rows[name][j]
            if row['index'] != int(index) or row['cid'] != refs[j]['cid']:
                raise ValueError(f'index or reference CID mismatch: {name}/{index}')
            if row['variant'] != name:
                raise ValueError('variant metadata mismatch')
            q = row['quality']
            if len(q['lpips_alex_per_block']) != 10 or not np.isfinite(
                    [q['psnr_db'], q['ssim'], q['lpips_alex'], *q['lpips_alex_per_block']]).all():
                raise ValueError(f'invalid quality metrics: {name}/{index}')
        keys = ('cid', 'seed', 'initial_noise_sha256', 'actions_sha256')
        if any(len({rows[n][j][k] for n in NAMES}) != 1 for k in keys):
            raise ValueError(f'conditioning mismatch at {index}')
        if rows['nano_last'][j]['latent_sha256'] != rows['nano_last_feedback_diagnostic'][j]['latent_sha256']:
            raise ValueError(f'shadow branch changed final latent at{index}')
        a = rows['nano_last_feedback_diagnostic'][j]['quality']['lpips_alex_per_block'][0]
        b = rows['nano_last_draft_history'][j]['quality']['lpips_alex_per_block'][0]
        if a != b:
            raise ValueError(f'history arms differ in first chunk at{index}')
    cids = [r['cid'] for r in refs]
    draws, sessions = cluster_draws(cids, rgb_repeats, seed)
    fvd_draws, _ = cluster_draws(cids, fvd_repeats, seed+1)
    features = {}
    for name in ('reference', *NAMES):
        features[name] = {kind:np.stack([np.load(root/name/i/'features.npz')[kind] for i in ids])
                          for kind in ('full121', 'uniform16')}
        if not all(np.isfinite(x).all() for x in features[name].values()):
            raise ValueError(f'nonfinite I3D features: {name}')
    summary = {'matched_count':len(ids), 'matched_indices':ids, 'session_count':sessions,
               'completed_per_variant':{n:len(sets[n]) for n in NAMES},
               'fvd_status':'full validation', 'variants':{},
               'all_noise_actions_matched':True, 'all_shadow_final_latents_equal':True}
    for name in NAMES:
        rs = rows[name]
        q = {k:float(np.mean([r['quality'][k] for r in rs])) for k in ('psnr_db','ssim','lpips_alex')}
        q['lpips_per_block'] = np.mean([r['quality']['lpips_alex_per_block'] for r in rs], axis=0).tolist()
        q['fps_aggregate'] = 120*len(rs)/sum(r['timing']['combined_s'] for r in rs)
        q['diagnostic_speed'] = name in ('nano_last_feedback_diagnostic','nano_last_draft_history')
        for kind in ('full121','uniform16'):
            q['fvd_'+kind] = fvd_low_rank(features[name][kind],features['reference'][kind])
        summary['variants'][name] = q
    pairs = {'nano_first_vs_last':('nano_first','nano_last'),
             'history_final_vs_draft':('nano_last_feedback_diagnostic','nano_last_draft_history')}
    comparisons = {}
    for label,(candidate,baseline) in pairs.items():
        cmp = {'candidate':candidate,'baseline':baseline,'rgb':{},'fvd':{}}
        for metric in ('psnr_db','ssim','lpips_alex'):
            a = [r['quality'][metric] for r in rows[candidate]]
            b = [r['quality'][metric] for r in rows[baseline]]
            cmp['rgb'][metric] = paired_delta(a,b,draws,metric=='lpips_alex')
        a = np.asarray([r['quality']['lpips_alex_per_block'] for r in rows[candidate]])
        b = np.asarray([r['quality']['lpips_alex_per_block'] for r in rows[baseline]])
        cmp['lpips_per_chunk_delta'] = paired_delta(a,b,draws)
        cmp['lpips_chunks2_to10'] = paired_delta(a[:,1:].mean(1),b[:,1:].mean(1),draws)
        cmp['lpips_chunks8_to10'] = paired_delta(a[:,7:].mean(1),b[:,7:].mean(1),draws)
        for kind in ('full121','uniform16'):
            a, b, real = features[candidate][kind],features[baseline][kind],features['reference'][kind]
            values = [fvd_low_rank(a[index],real[index])-fvd_low_rank(b[index],real[index]) for index in fvd_draws]
            cmp['fvd'][kind] = {'candidate_minus_baseline':summary['variants'][candidate]['fvd_'+kind]-summary['variants'][baseline]['fvd_'+kind],
                                'ci95':np.quantile(values,[.025,.975]).tolist(), 'lower_is_better':True}
        comparisons[label] = cmp
    feedback = comparisons['history_final_vs_draft']['rgb']['lpips_alex']
    summary['final_history_vs_draft_lpips_wins'] = feedback['wins']
    summary['final_history_vs_draft_count'] = len(ids)
    report = {'count':len(ids),'sessions':sessions,'seed':seed,
              'rgb_bootstrap_repeats':rgb_repeats,'fvd_bootstrap_repeats':fvd_repeats,
              'resampling':'paired source-session cluster bootstrap; equal clip weight within each resample',
              'scope':'descriptive comparison on reused validation; not an independent generalization guarantee; chunk intervals are pointwise, without multiple-comparison correction',
              'comparisons':comparisons}
    per_clip = [{'index':int(index), 'session':session_key(cids[j]),
                 'variants':{n:rows[n][j]['quality'] for n in NAMES}} for j,index in enumerate(ids)]
    for filename,obj in [('validated-summary.json',summary),('paired-analysis.json',report),('per-clip-quality.json',per_clip)]:
        tmp = root/(filename+'.tmp')
        tmp.write_text(json.dumps(obj,indent=2)+'\n')
        tmp.replace(root/filename)
    return report


def numerical_check():
    rng = np.random.default_rng(7)
    x,y = rng.normal(size=(12,20)),rng.normal(size=(12,20))
    assert abs(fvd_low_rank(x,y)-fvd(x,y)) < 1e-5
    assert fvd_low_rank(x,x) < 1e-10
    assert abs(fvd_low_rank(x,x+1)-20) < 1e-10
    assert abs(fvd_low_rank(x,y)-fvd_low_rank(y,x)) < 1e-10


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path,nargs='?')
    p.add_argument('--rgb-repeats',type=int,default=5000)
    p.add_argument('--fvd-repeats',type=int,default=512)
    a=p.parse_args()
    numerical_check()
    if a.root:
        report=analyze(a.root,a.rgb_repeats,a.fvd_repeats)
        print(json.dumps({'count':report['count'],'sessions':report['sessions'],'saved':'validated-summary.json and paired-analysis.json'}))
    else:
        print('Low-rank FVD identity, translation, symmetry and covariance-reference checks passed.')
