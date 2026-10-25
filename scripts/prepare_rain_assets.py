"""Create the rain comparison viewing copies with the report's model names."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('source', type=Path, help='Existing rain-review directory')
a = p.parse_args()
r = Path(__file__).resolve().parents[1]
assets = r / 'docs/assets'
ff = '/opt/homebrew/bin/ffmpeg'
title_font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 23)
small_font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 16)
clips = [('2p5mps', '直行 2.5 m/s'), ('0p5mps', '慢行 0.5 m/s'), ('stop', '静止 0 m/s')]
for tag, action in clips:
    source = a.source / f'rain-three-models-{tag}.mp4'
    dest = assets / f'rain-comparison-{tag}.mp4'
    with tempfile.TemporaryDirectory() as tmp:
        labels = Path(tmp) / 'labels.png'
        im = Image.new('RGBA', (2496, 48), '#f7f7f5')
        d = ImageDraw.Draw(im)
        for i, name in enumerate(['旧 stage3 student', '4222 clean', '4222+noisy kv reuse']):
            x = i * 832
            d.text((x + 12, -2), name, font=title_font, fill='#37352f')
            d.text((x + 12, 26), f'{action} · seed 2026 · TinyVAE BF16 流式', font=small_font, fill='#787774')
        im.save(labels)
        subprocess.run([ff, '-y', '-v', 'error', '-filter_complex_threads', '1',
                        '-i', str(source), '-loop', '1', '-framerate', '20', '-i', str(labels),
                        '-filter_complex', '[0:v][1:v]overlay=0:0:shortest=1[v]', '-map', '[v]',
                        '-an', '-frames:v', '121', '-r', '20', '-c:v', 'libx264', '-crf', '23',
                        '-preset', 'fast', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(dest)], check=True)
    subprocess.run([ff, '-y', '-v', 'error', '-ss', '3', '-i', str(dest), '-frames:v', '1',
                    '-q:v', '3', str(assets / f'rain-poster-{tag}.jpg')], check=True)
    print(json.dumps({'file': dest.name, 'bytes': dest.stat().st_size}))
