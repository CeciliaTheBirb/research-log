"""Replace all model headers with hardware-independent optimized-runtime labels."""
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

r = Path(__file__).resolve().parents[1]
video = r / 'docs/assets/model-comparison.mp4'
font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 27)
small = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 20)
with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    im = Image.new('RGBA', (1664, 1136), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    panels = [('真实参考视频', 'Ground truth'),
              ('旧 stage3 student', '原生单卡＋文本缓存 | 每块 4 步'),
              ('4222 clean', '直接单卡＋文本缓存 | 首块 4 步，后续 2 步'),
              ('4222+noisy kv reuse', '直接单卡＋文本缓存 | 复用后续含噪 KV')]
    for k, (title, subtitle) in enumerate(panels):
        x, y = k % 2 * 832, k // 2 * 536
        d.rectangle((x, y, x + 831, y + 55), fill=(17, 25, 36, 255))
        d.text((x + 12, y), title, font=font, fill='white')
        d.text((x + 12, y + 31), subtitle, font=small, fill=(170, 212, 239))
    d.rectangle((0, 1072, 1663, 1135), fill=(17, 25, 36, 255))
    d.text((12, 1076), '片段顺序：0000 → 0057 → 0114 | 全部按 20 FPS 同步播放', font=font, fill='white')
    d.text((12, 1110), '优化后的部署路径；生成吞吐请查看网页的 L40S / A5000 结果表', font=small, fill=(190, 200, 214))
    im.save(t / 'headers.png')
    subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error', '-filter_complex_threads', '1',
        '-i', str(video), '-loop', '1', '-framerate', '20', '-i', str(t / 'headers.png'),
        '-filter_complex', '[0:v][1:v]overlay=0:0:shortest=1[v]', '-map', '[v]', '-an',
        '-frames:v', '363', '-r', '20', '-c:v', 'libx264', '-crf', '24', '-preset', 'fast',
        '-maxrate', '8M', '-bufsize', '16M', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(t / 'new.mp4')], check=True)
    video.write_bytes((t / 'new.mp4').read_bytes())
for i, sec in enumerate([3, 9.05, 15.1]):
    target = r / 'docs/assets' / ('quality-poster.jpg' if i == 0 else f'quality-poster-{i}.jpg')
    subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error', '-ss', str(sec), '-i', str(video),
                    '-frames:v', '1', '-q:v', '3', str(target)], check=True)
print('Optimized-only video prepared', video.stat().st_size)
