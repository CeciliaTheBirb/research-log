"""Plot full-validation paired LPIPS effects across generated chunks."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'data/2026-10-29/edge-nano-4444/paired-analysis.json').read_text())
assert data['count'] == 172
out = root / 'docs/assets/edge-nano-20261029'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, key, title, color in zip(
        axes, ['nano_first_vs_last', 'history_final_vs_draft'],
        ['Nano first - Nano last', 'Final history - Edge draft history'],
        ['#497594', '#458373']):
    delta = data['comparisons'][key]['lpips_per_chunk_delta']
    x = np.arange(1, 11)
    low, high = np.asarray(delta['ci95'])
    ax.axhline(0, color='#777777', linewidth=.8, linestyle='--')
    ax.fill_between(x, low, high, color=color, alpha=.18, linewidth=0)
    ax.plot(x, delta['mean_candidate_minus_baseline'], color=color,
            marker='o', markersize=4, linewidth=1.8)
    ax.set_title(title, loc='left', fontweight='semibold', pad=13)
    ax.set_xlabel('Generated chunk (12 RGB frames each)')
    ax.set_xticks(x)
    ax.set_xlim(.8, 10.2)
    ax.grid(axis='y', color='#eeeeee')
    ax.set_axisbelow(True)
axes[0].set_ylabel('LPIPS difference (negative favors first option)')
fig.text(.08, .035,
         '172 validation clips / 170 sessions; pointwise 95% paired session-bootstrap CI, 5,000 draws.',
         color='#666666', fontsize=9)
fig.subplots_adjust(left=.09, right=.985, bottom=.24, top=.86, wspace=.13)
for extension in ('svg', 'png'):
    path = out / f'paired-lpips-by-chunk.{extension}'
    fig.savefig(path, dpi=180,
                facecolor='white', metadata={'Date': None} if extension == 'svg' else None)
    if extension == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
plt.close(fig)
print('Saved full172 LPIPS chunk plots.')
