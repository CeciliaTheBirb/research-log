"""Stitch matching reference/old/noisy1:1/noisy+GAN evaluation clips without changing playback speed."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--input-root', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
destination = repo / 'docs/assets/noisy-gan-comparison.mp4'
ffmpeg = shutil.which('ffmpeg')
assert ffmpeg
font_path = '/System/Library/Fonts/PingFang.ttc'
font = ImageFont.truetype(font_path, 27)
small = ImageFont.truetype(font_path, 20)
provenance = []
with tempfile.TemporaryDirectory() as temporary:
    temp = Path(temporary)
    headers = Image.new('RGBA', (1664, 1136), (0, 0, 0, 0))
    draw = ImageDraw.Draw(headers)
    titles = [('真实参考视频', 'Ground truth'), ('旧 stage3 student', '原生每块4步'),
              ('4222+noisy kv reuse', 'student:fake = 1:1 | 800次 student'),
              ('4222+noisy kv reuse + video GAN', 'student:fake = 1:1 | λGAN = 0.001')]
    for index, (title, subtitle) in enumerate(titles):
        x, y = index % 2 * 832, index // 2 * 536
        draw.rectangle((x, y, x + 831, y + 55), fill=(24, 27, 32, 255))
        draw.text((x + 12, y), title, font=font, fill='white')
        draw.text((x + 12, y + 31), subtitle, font=small, fill=(195, 210, 222))
    draw.rectangle((0, 1072, 1663, 1135), fill=(24, 27, 32, 255))
    draw.text((12, 1076), '片段顺序：0000 → 0057 → 0114 | 统一按20 FPS播放', font=font, fill='white')
    draw.text((12, 1110), 'TinyVAE BF16流式 | 同一首帧、动作和初始噪声 | 质量指标取未压缩RGB', font=small, fill=(195, 210, 222))
    headers.save(temp / 'headers.png')
    outputs = []
    for index in (0, 57, 114):
        paths = [args.input_root / name / f'{index:04d}' / filename for name, filename in [
            ('old4444_native', 'reference.mp4'), ('old4444_native', 'generated.mp4'),
            ('noisy_matched1600_nocaption', 'generated.mp4'), ('noisy_gan1600_nocaption', 'generated.mp4')]]
        assert all(path.is_file() for path in paths)
        panel = temp / f'{index:04d}.mp4'
        filters = [f'[{i}:v]setpts=PTS-STARTPTS,pad=832:536:0:56:black[p{i}]' for i in range(4)]
        filters += ['[p0][p1][p2][p3]xstack=inputs=4:layout=0_0|832_0|0_536|832_536:shortest=1,pad=1664:1136:0:0:black[stack]',
                    '[stack][4:v]overlay=0:0:shortest=1[v]']
        command = [ffmpeg, '-y', '-v', 'error', '-filter_complex_threads', '1']
        for path in paths:
            command += ['-i', str(path)]
        command += ['-loop', '1', '-framerate', '20', '-i', str(temp / 'headers.png'),
                    '-filter_complex', ';'.join(filters), '-map', '[v]', '-frames:v', '121',
                    '-an', '-r', '20', '-c:v', 'libx264', '-crf', '24', '-preset', 'fast',
                    '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(panel)]
        subprocess.run(command, check=True)
        outputs.append(panel)
        provenance.append({'index': index, 'inputs': [{'path': str(p.relative_to(args.input_root)),
                           'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]})
    (temp / 'concat.txt').write_text(''.join(f"file '{p}'\n" for p in outputs))
    subprocess.run([ffmpeg, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(temp / 'concat.txt'),
                    '-c', 'copy', '-movflags', '+faststart', str(destination)], check=True)
    subprocess.run([ffmpeg, '-y', '-v', 'error', '-ss', '3', '-i', str(destination), '-frames:v', '1', '-q:v', '3',
                    str(destination.with_suffix('.jpg'))], check=True)
data = {'output': 'docs/assets/' + destination.name, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'frames': 363, 'fps': 20, 'duration_s': 18.15, 'panels': titles, 'clips': provenance}
(repo / 'data/2026-10-26/noisy-conditional-gan/video-provenance.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print(destination, destination.stat().st_size)
