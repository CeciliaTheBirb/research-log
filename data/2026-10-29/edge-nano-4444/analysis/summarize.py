"""Aggregate matched native features; render compact, synchronized video grids."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

NAMES = ('edge4444','nano4444','nano_first','nano_last',
         'nano_last_feedback_diagnostic','nano_last_draft_history')


def fvd(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    a, b = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    eig, u = np.linalg.eigh((a+a.T)*.5)
    root = (u*np.sqrt(np.maximum(eig,0)))@u.T
    c = root@b@root
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh((c+c.T)*.5),0)).sum()
    return max(0.,float(np.sum((x.mean(0)-y.mean(0))**2)+np.trace(a)+np.trace(b)-2*cross))


def summarize(output):
    output = Path(output)
    counts = {name: sorted(p.parent.name for p in (output/name).glob('*/metrics.json')) for name in NAMES}
    common = sorted(set.intersection(*(set(v) for v in counts.values())))
    report = {'completed_per_variant':{k:len(v) for k,v in counts.items()}, 'matched_count':len(common),
              'fvd_status':'full validation' if len(common)==172 else 'pending: do not interpret pilot FVD',
              'variants':{}}
    for name in NAMES:
        rows = [json.loads((output/name/i/'metrics.json').read_text()) for i in common]
        if not rows:
            continue
        result = {k:float(np.mean([r['quality'][k] for r in rows])) for k in ('psnr_db','ssim','lpips_alex')}
        result['lpips_per_block'] = np.mean([r['quality']['lpips_alex_per_block'] for r in rows],axis=0).tolist()
        result['fps_aggregate'] = 120*len(rows)/sum(r['timing']['combined_s'] for r in rows)
        result['diagnostic_speed'] = 'diagnostic' in name or 'draft_history' in name
        if len(common)==172:
            for feature in ('full121','uniform16'):
                pred = np.stack([np.load(output/name/i/'features.npz')[feature] for i in common])
                real = np.stack([np.load(output/'reference'/i/'features.npz')[feature] for i in common])
                result['fvd_'+feature] = fvd(pred,real)
        report['variants'][name] = result
    tmp = output/'summary.json.tmp'
    tmp.write_text(json.dumps(report,indent=2)+'\n')
    tmp.replace(output/'summary.json')
    return report


def stitch(output, index):
    output = Path(output)
    panels = [('Reference',output/'reference'/f'{index:04d}'/'reference.mp4'),
              ('Edge 4444',output/'edge4444'/f'{index:04d}'/'generated.mp4'),
              ('Nano 4444',output/'nano4444'/f'{index:04d}'/'generated.mp4'),
              ('Nano first',output/'nano_first'/f'{index:04d}'/'generated.mp4'),
              ('Nano last + final history',output/'nano_last'/f'{index:04d}'/'generated.mp4'),
              ('Nano last + Edge draft history',output/'nano_last_draft_history'/f'{index:04d}'/'generated.mp4')]
    if not all(p.is_file() for _,p in panels):
        return
    out = output/'comparisons'
    out.mkdir(exist_ok=True)
    args = ['ffmpeg','-v','error','-y','-filter_complex_threads','1']
    for _,p in panels:
        args += ['-i',str(p)]
    filters = [f"[{i}:v]scale=416:240,pad=416:268:0:28:color=white,drawtext=text='{name}':fontcolor=black:fontsize=15:x=8:y=6[p{i}]"
               for i,(name,_) in enumerate(panels)]
    filters += [''.join(f'[p{i}]' for i in range(6))+'xstack=inputs=6:layout=0_0|416_0|832_0|0_268|416_268|832_268[v]']
    args += ['-filter_complex',';'.join(filters),'-map','[v]','-an','-c:v','libx264','-threads','2','-crf','20',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'comparison-{index:04d}.mp4')]
    subprocess.run(args,check=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.output),indent=2))
    for i in (0,57,114):
        stitch(a.output,i)
