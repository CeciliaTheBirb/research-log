"""Four-model synchronized viewing copies, without deriving quality from MP4."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, required=True)
args = parser.parse_args()
workspace = args.workspace.resolve()
root = workspace/'outputs/hf-hybrid-20261025'
out = root/'stitched'
out.mkdir(parents=True, exist_ok=True)
ff = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
probe = shutil.which('ffprobe') or '/opt/homebrew/bin/ffprobe'
font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 26)
small = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 20)
panels = [('旧 stage3 student','4,4,4,4,… · 原生单卡＋文本缓存'),
          ('4222 clean','4,2,2,2,… · clean KV'),
          ('4222+noisy kv reuse','4,2,2,2,… · noisy KV reuse'),
          ('43222+hybrid forcing','4,3,2,2,… · sink 1 / history 2 ＋ 历史记忆')]
audit = []

def stitch(sources, target, note):
    for source in sources:
        assert source.is_file(), source
        meta = json.loads(subprocess.check_output([probe,'-v','error','-show_streams','-of','json',str(source)]))['streams'][0]
        assert (meta['width'],meta['height'],meta['nb_frames'],meta['r_frame_rate']) == (832,480,'121','20/1'), (source,meta)
    with tempfile.TemporaryDirectory() as tmp:
        overlay = Image.new('RGBA',(1664,1136),(0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        for i,(title,subtitle) in enumerate(panels):
            x,y = i%2*832,i//2*536
            draw.rectangle((x,y,x+831,y+55),fill='#f7f7f5')
            draw.text((x+12,y-1),title,font=font,fill='#37352f')
            draw.text((x+12,y+31),subtitle,font=small,fill='#787774')
        draw.rectangle((0,1072,1663,1135),fill='#f7f7f5')
        draw.text((12,1076),note,font=font,fill='#37352f')
        draw.text((12,1110),'相同输入与噪声 · 121 帧 / 20 FPS 同步播放 · TinyVAE BF16 流式',font=small,fill='#787774')
        labels = Path(tmp)/'labels.png'; overlay.save(labels)
        command = [ff,'-y','-v','error','-filter_complex_threads','1']
        for source in sources: command += ['-i',str(source)]
        command += ['-loop','1','-framerate','20','-i',str(labels),'-filter_complex',
                    '[0:v]pad=832:536:0:56:white[a];[1:v]pad=832:536:0:56:white[b];'
                    '[2:v]pad=832:536:0:56:white[c];[3:v]pad=832:536:0:56:white[d];'
                    '[a][b][c][d]xstack=inputs=4:layout=0_0|832_0|0_536|832_536, pad=1664:1136:0:0:white[grid];'
                    '[grid][4:v]overlay=shortest=1[v]', '-map','[v]','-an','-frames:v','121','-r','20',
                    '-c:v','libx264','-crf','24','-preset','fast','-threads','2','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(target)]
        subprocess.run(command,check=True)
    subprocess.run([ff,'-y','-v','error','-ss','3','-i',str(target),'-frames:v','1','-q:v','3',str(target.with_suffix('.jpg'))],check=True)
    audit.append({'output':str(target.relative_to(workspace)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                  'inputs':[{'path':str(s.relative_to(workspace)),'sha256':hashlib.sha256(s.read_bytes()).hexdigest()} for s in sources]})
    print(target.name,flush=True)

for index in [0,57,114]:
    name=f'{index:04d}'; base=workspace/'outputs/inference-speed-20261025'
    sources=[base/'old4444_native'/name/'generated.mp4',
             base/'single_device_text_cache/clean3200_nocaption'/name/'generated.mp4',
             base/'single_device_text_cache/noisy_matched1600_nocaption'/name/'generated.mp4',
             root/'benchmark-a5000/hybrid2400_nocaption'/name/'generated.mp4']
    stitch(sources,out/f'four-models-{name}.mp4',f'验证片段 {name} · 四模型同步对比')
concat = out/'concat.txt'
concat.write_text(''.join(f"file '{out/f'four-models-{i:04d}.mp4'}'\n" for i in [0,57,114]))
subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy',
                '-movflags','+faststart',str(out/'four-models-all.mp4')],check=True)
for tag,folder,action in [('2p5mps','rain-regression-20261024','直行 2.5 m/s'),
                          ('0p5mps','rain-slow-regression-20261024','慢行 0.5 m/s'),
                          ('stop','rain-stop-regression-20261024','静止')]:
    base=workspace/'outputs'/folder/'rain'
    sources=[base/'old4444_bf16_stream/generated.mp4',base/'clean3200/generated.mp4',
             base/'noisy_matched1600/generated.mp4',root/'rain'/tag/'generated.mp4']
    if sources[-1].exists():
        stitch(sources,out/f'rain-four-models-{tag}.mp4',f'雨后路口 · {action}')
(out/'provenance.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
