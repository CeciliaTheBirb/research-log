"""Verify matched inputs and assemble Edge/Nano/Super comparison videos."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont
from native_follow_cases import expected_cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args=parser.parse_args()
    root=args.root; manifest=json.loads(args.manifest.read_text())
    expected=expected_cases(manifest)
    models=['edge','nano','super']
    for model in models:
        status=json.loads((root/model/'complete.json').read_text())
        assert status['complete'] and status['records']==expected and not status['limited'], status
    records=[]
    for path in sorted((root/'edge').rglob('result.json')):
        relative=path.relative_to(root/'edge')
        rows=[json.loads((root/model/relative).read_text()) for model in models]
        for key in ['anchor_sha256','initial_noise_sha256','raw_actions_sha256','raw_actions_shape','prompt','conditioning','manifest_sha256']:
            assert all(row[key]==rows[0][key] for row in rows), (relative,key)
        assert all(row['prompt_truncated'] is False for row in rows)
        records.append((relative,rows))
    assert len(records)==expected
    for model in models:
        for anchor in manifest['anchors']:
            for seed in manifest['seeds']:
                folder=root/model/anchor['id']/f'seed-{seed}'
                if manifest.get('suite')=='forward_encounter':
                    encounters=[json.loads((folder/c['id']/'result.json').read_text()) for c in manifest['cases']]
                    assert len({x['raw_actions_sha256'] for x in encounters})==1, 'Encounter changed numeric forward motion'
                    assert len({x['prompt_ids_sha256'] for x in encounters})==3, 'Encounter prompts are not distinct'
                    continue
                controls=[json.loads((folder/c/'result.json').read_text()) for c in ['fd_stop_control','fd_forward','fd_left','fd_right']]
                assert len({x['prompt_ids_sha256'] for x in controls})==1, 'Action intervention changed text'
                stops=[json.loads((folder/c/'result.json').read_text()) for c in ['fd_stop_control','fd_stop_person_ltr','fd_stop_person_rtl']]
                assert len({x['raw_actions_sha256'] for x in stops})==1, 'Text intervention changed actions'
    out=root/'stitched';out.mkdir(exist_ok=True)
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    font_path=next((p for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf'] if Path(p).exists()),None)
    font=ImageFont.truetype(font_path,25) if font_path else ImageFont.load_default()
    small=ImageFont.truetype(font_path,19) if font_path else font
    width,height,header,footer=832,480,64,104
    title_map={'fd_stop_control':'Robot stopped / no new foreground pedestrian',
               'fd_forward_control':'Robot advances 0.5 m/s / no new foreground pedestrian',
               'fd_forward_oncoming_left':'Robot advances 0.5 m/s / oncoming pedestrian passes on ROBOT LEFT',
               'fd_forward_oncoming_right':'Robot advances 0.5 m/s / oncoming pedestrian passes on ROBOT RIGHT',
               'fd_stop_person_ltr':'Robot stopped / pedestrian crosses LEFT to RIGHT',
               'fd_stop_person_rtl':'Robot stopped / pedestrian crosses RIGHT to LEFT',
               'fd_forward':'Numeric action: forward 0.5 m/s / fixed neutral text',
               'fd_left':'Numeric action: left turn +10 deg/s, 0.5 m/s / fixed neutral text',
               'fd_right':'Numeric action: right turn -10 deg/s, 0.5 m/s / fixed neutral text',
               'i2v_control':'I2V: no new foreground pedestrian / NO numeric actions',
               'i2v_person_ltr':'I2V: pedestrian crosses LEFT to RIGHT / NO numeric actions',
               'i2v_person_rtl':'I2V: pedestrian crosses RIGHT to LEFT / NO numeric actions'}
    rendered=[];groups={}
    for relative, rows in records:
        example=rows[0]; case=example['case']['id']; anchor=example['anchor'];seed=example['seed']
        stem=f'{anchor}-seed{seed}-{case}'
        overlay=Image.new('RGBA',(width*3,height+header+footer),(0,0,0,0))
        draw=ImageDraw.Draw(overlay)
        for k,model in enumerate(models):
            x=k*width
            draw.rectangle((x,0,x+width-1,header-1),fill=(25,31,40,255))
            draw.text((x+14,4),f'Native Cosmos3-{model.title()} / NO adapter',font=font,fill='white')
            draw.text((x+14,36),f'{rows[k]["gpu_count"]} GPU | BF16 | TinyVAE streaming',font=small,fill=(192,212,228))
        draw.rectangle((0,height+header,width*3,height+header+footer),fill=(25,31,40,255))
        draw.text((14,height+header+8),title_map[case],font=font,fill='white')
        draw.text((14,height+header+44),f'{anchor} | seed {seed} | 121 frames / 20 FPS | identical image, anchor, initial noise and conditions across models',font=small,fill=(192,212,228))
        draw.text((14,height+header+74),'Small controllability pilot; no paired pixel metrics or population FVD. Left/right are image coordinates. Identical playback speed.',font=small,fill=(192,212,228))
        label=out/f'{stem}-labels.png'; overlay.save(label)
        target=out/f'{stem}.mp4'
        command=[ffmpeg,'-y','-v','error','-filter_complex_threads','1']
        for model in models:
            command+=['-i',str(root/model/relative.parent/'generated.mp4')]
        command+=['-loop','1','-framerate','20','-i',str(label)]
        graph=';'.join(f'[{i}:v]setpts=PTS-STARTPTS,pad={width}:{height+header}:0:{header}:color=0x191f28[v{i}]' for i in range(3))
        graph+=f';[v0][v1][v2]hstack=inputs=3[grid];[grid]pad={width*3}:{height+header+footer}:0:0:color=0x191f28[canvas];[canvas][3:v]overlay=shortest=1[v]'
        command+=['-filter_complex',graph,'-map','[v]','-frames:v','121','-r','20','-an','-c:v','libx264','-crf','19',
                  '-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        subprocess.run(command,check=True)
        group=f'{anchor}-seed{seed}-'+('i2v' if case.startswith('i2v') else 'joint-fd')
        groups.setdefault(group,[]).append((case,target))
        rendered.append({'file':target.name,'case':case,'anchor':anchor,'seed':seed,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    bundles=[]
    order={c['id']:i for i,c in enumerate(manifest['cases'])}
    for name, parts in groups.items():
        parts.sort(key=lambda item:order[item[0]])
        listing=out/f'{name}.txt'
        listing.write_text(''.join(f"file '{path.name}'\n" for _,path in parts))
        target=out/f'{name}.mp4'
        subprocess.run([ffmpeg,'-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],check=True)
        poster=target.with_suffix('.jpg')
        subprocess.run([ffmpeg,'-y','-v','error','-ss','2','-i',str(target),'-frames:v','1',str(poster)],check=True)
        bundles.append(dict(file=target.name,poster=poster.name,cases=[x[0] for x in parts],
                            frames=121*len(parts),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    audit=dict(complete=True,models=models,matched_cases=expected,matched_input_checks_passed=True,
               action_interventions_have_identical_text=None if manifest.get('suite')=='forward_encounter' else True,text_interventions_have_identical_actions=True,
               videos=rendered,bundles=bundles,scope=manifest['scope'])
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    cards=''.join(f'<article><h2>{html.escape(x["file"])}</h2><video controls preload="metadata" poster="{x["poster"]}" src="{x["file"]}"></video><p>{html.escape(" → ".join(x["cases"]))}</p></article>' for x in bundles)
    intro = ("每个画面依次为 Edge、Nano、Super。均无训练 adapter、50 步、BF16 与 TinyVAE 流式解码。"
             "FD 组固定动作改文本、固定文本改动作；I2V 组不输入数值动作。")
    if manifest.get('suite') == 'forward_encounter':
        intro = ("每个画面依次为 Edge、Nano、Super。机器人三种条件均由相同数值动作控制，以 0.5 m/s 向前直行，共 3 米。"
                 "每组依次为无人迎面、行人迎面走来从机器人左侧擦身、从右侧擦身。左右以机器人摄像头为准。"
                 "三模型均无训练 adapter、50 步、BF16 与 TinyVAE 流式解码。")
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生 Super / Nano / Edge 控制对照</title><style>body{font:17px system-ui;max-width:1500px;margin:40px auto;padding:20px;color:#242424}video{width:100%}article{margin:50px 0}p{line-height:1.7}</style><h1>原生 Super / Nano / Edge 控制对照</h1><p>'+intro+'不同模型的 GPU 数不同，推理时间不作为同卡速度排名。这是两种场景、两个种子的控制小实验，暂不估计总体 FVD；FVD本身不要求逐条配对。反事实事件与动作没有对应真实未来，无法计算这组展示的配对 PSNR / SSIM。此前172条验证集指标单独保留。</p>'+cards+'</html>')
    print(json.dumps(dict(complete=True,bundles=len(bundles),output=str(out))),flush=True)


if __name__=='__main__':
    main()
