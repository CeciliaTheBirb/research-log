"""Isolate official Hybrid Forcing window/readout and structured-prompt settings.

Uses preselected validation indices, identical cached anchor/actions/noise, and
the pinned native runtime. This is a diagnostic subset, never a replacement FVD.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file

from evaluate_hf_hybrid import mapped


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--indices', nargs='+', type=int, default=[0,19,38,57,76,95,114,133,152,171])
    args = p.parse_args()
    manifest = json.loads(mapped(args.manifest).read_text())
    root = mapped(args.output)
    assert root.is_absolute()
    root.mkdir(parents=True, exist_ok=True)
    runtime = mapped(manifest['runtime'])
    sys.path.insert(0, str(runtime))
    from inference import load_model, read_model_config, read_generation_defaults
    from navworld_inference.tinyvae_checkpoint import load_tinyvae_decoder
    from navworld_inference import structured_prompt as sp
    from navworld_inference.vendored.cosmos3 import transformer_cosmos3 as native_attn
    from evaluate_stage3 import Quality, reference_frames, save_json, tiny_decode, uint8_frames
    from costream.config import load_config
    from costream.data import ClipDataset
    from costream.sampling import write_mp4

    cfg = load_config('configs/stage2_edge_navworld_422_clean.yaml', ['data.stage=false'])
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    checkpoint = mapped(manifest['hybrid_checkpoint'])
    params, filters = read_model_config(checkpoint/'inference_config.yaml', mapped(cfg.model.root))
    model, loaded = load_model(params, checkpoint, filters, device,
        checkpoint/'action_weights.safetensors', checkpoint/'time_weights.safetensors',
        checkpoint/'history_weights.safetensors')
    model.vae.to('cpu')
    torch.cuda.empty_cache()
    assert loaded['history_tensors'] == 28
    assert model._raw_transformer().rotary_emb.inv_freq.dtype == torch.float32
    tiny = load_tinyvae_decoder(mapped(manifest['tinyvae']), device, dtype=torch.bfloat16)
    quality = Quality(mapped(manifest['assets']), device)
    dataset = ClipDataset(cfg.data.root, 'val', index=cfg.data.val_index, caption_mode='none')
    baseline = mapped(manifest['runtime']).parents[2]/'evaluation/hf-hybrid-20261025'
    assert baseline.is_dir(), baseline
    defaults = read_generation_defaults(checkpoint/'inference_config.yaml')
    dense = 'A forward-facing camera on an autonomous vehicle observes the surrounding street scene.'
    structured = sp.caption_to_prompt(sp.assemble_caption(
        sp.minimal_scene_draft(dense, sp.clip_time_range(121,20)), dense, sp.media_fields(121,20,480,832)))
    # Generic first-person context only: no validation caption or future image is read.
    variants = {
        'native_s1h2_empty': ('', dict(defaults)),
        'full_history_empty': ('', {**defaults, 'attention_sink_size':0, 'attention_history_cutoff':999}),
        'memory_disabled_empty': ('', {**defaults, 'history_memory_readout':False}),
        'native_s1h2_structured_generic': (structured, dict(defaults)),
    }
    save_json(root/'settings.json', {'indices':args.indices, 'job_id':os.environ.get('SLURM_JOB_ID'),
        'gpu':torch.cuda.get_device_name(), 'runtime_commit':manifest['runtime_commit'], 'loaded':loaded,
        'variants':{k:{'prompt':v[0],'generation':v[1]} for k,v in variants.items()},
        'scope':'Preselected diagnostic subset, no full-set FVD. Generic structured text contains no scene-specific or future information.'})
    layer_ids = {id(x.self_attn):i for i,x in enumerate(model._raw_transformer().layers)}
    native_readout = native_attn._add_history_readout
    native_forward = model._generation_forward
    readout_log, forward_log = [], []
    trace_enabled = False

    def watched_readout(attn, attention_out, q_pre, cos, sin, cache):
        result = native_readout(attn, attention_out, q_pre, cos, sin, cache)
        if trace_enabled and layer_ids[id(attn)] in (0,13,27):
            state = None if cache is None else cache.get('history')
            readout_log.append({'layer':layer_ids[id(attn)],
                'retained_blocks':[] if cache is None else [int(b['block_index']) for b in cache.get('blocks',())],
                'history_tokens':0 if state is None else int(state['count']),
                'readout_enabled':bool(attn.history_readout_enabled),
                'output_delta_rms':float((result.float()-attention_out.float()).square().mean().sqrt()),
                'attention_rms':float(attention_out.float().square().mean().sqrt())})
        return result

    def watched_forward(*a, **kw):
        if trace_enabled:
            caches = kw['generation_kv_caches']
            cache = caches[0]
            state = cache.get('history')
            forward_log.append({'block':int(kw['generation_cache_block_index']),
                'commit':bool(kw['update_generation_kv_cache']),
                'visible_blocks':[int(b['block_index']) for b in cache.get('blocks',())],
                'history_tokens':0 if state is None else int(state['count'])})
        return native_forward(*a, **kw)

    model._generation_forward = watched_forward
    records = []
    with patch.object(native_attn, '_add_history_readout', watched_readout):
        for i in args.indices:
            item, record = dataset[i], dataset.records[i]
            seed = manifest['seed']+i
            noise = load_file(str(mapped(manifest['initial_noise_dir'])/f'seed{seed}.safetensors'))['noise'].to(device)
            anchor = item['latent'][None,:,:1].to(device=device,dtype=torch.bfloat16)
            actions = item['actions'][1:].to(device)
            assert actions.shape == (120,9)
            real = reference_frames(record,mapped(manifest['reference_root']))
            original = json.loads((baseline/'hybrid2400_nocaption'/f'{i:04d}'/'metrics.json').read_text())
            prefix_hash = None
            for name,(prompt,generation) in variants.items():
                if name == 'native_s1h2_empty' and i != args.indices[0]:
                    records.append({'variant':name,'index':i,'quality':original['quality'],'reused_baseline':True})
                    continue
                folder=root/name/f'{i:04d}'
                folder.mkdir(parents=True,exist_ok=True)
                readout_log.clear(); forward_log.clear()
                trace_enabled = i == args.indices[0]
                draws=[]
                def fixed_noise(shape, **kw):
                    assert tuple(shape)==(1,48,3,30,52) and len(draws)<10
                    start=1+3*len(draws); draws.append(start)
                    return noise[:,:,start:start+3].to(device=kw['device'],dtype=kw['dtype']).clone()
                with patch.object(torch,'randn',fixed_noise):
                    latent=model.generate(prompt=prompt,negative_prompt='',num_frames=121,height=480,width=832,
                        num_inference_steps=4,guidance_scale=1.,flow_shift=5.,first_frame_latent=anchor,
                        camera_actions=actions,seed=seed,decode_output=False,**generation)
                assert len(draws)==10
                metadata=copy.deepcopy(model.generation_metadata)
                assert metadata['denoise_call_count']==23 and metadata['cache_commit_count']==11
                current_prefix_hash=hashlib.sha256(latent[:,:,:10].float().cpu().numpy().tobytes()).hexdigest()
                if prefix_hash is None: prefix_hash=current_prefix_hash
                elif not prompt: assert current_prefix_hash==prefix_hash, 'Difference before first eviction'
                frames=uint8_frames(tiny_decode(tiny,latent,False))
                rgb_hash=hashlib.sha256(frames.tobytes()).hexdigest()
                if name=='native_s1h2_empty':
                    assert rgb_hash==original['timing']['rgb_sha256'], 'Native audited repeat differs from completed evaluation'
                paired=quality.paired(frames,real)
                result={'variant':name,'index':i,'seed':seed,'quality':paired,'generation':metadata,
                    'rgb_sha256':rgb_hash,'prefix_latent_sha256':current_prefix_hash,
                    'history_readout_trace':copy.deepcopy(readout_log),'window_trace':copy.deepcopy(forward_log),
                    'prompt_tokens':len(model.build_prompt_ids(prompt,121,480,832))}
                save_json(folder/'metrics.json',result)
                if i in (0,57,114):
                    write_mp4(frames,folder/'generated.mp4',20)
                records.append(result)
                print(json.dumps({'variant':name,'index':i,'lpips':paired['lpips_alex'],'psnr':paired['psnr_db']}),flush=True)
                save_json(root/'progress.json',{'records':records,'complete':False})
    summary={}
    for name in variants:
        rows=[x for x in records if x['variant']==name]
        assert len(rows)==len(args.indices)
        summary[name]={k:float(np.mean([x['quality'][k] for x in rows])) for k in ['lpips_alex','psnr_db','ssim']}
        for k in ['lpips_alex_per_block','psnr_db_per_block','ssim_per_block']:
            summary[name][k]=np.mean([x['quality'][k] for x in rows],axis=0).tolist()
    save_json(root/'summary.json',{'complete':True,'indices':args.indices,'count':len(args.indices),'variants':summary,'records':records})
    print(json.dumps({'complete':True,'summary':summary}),flush=True)


if __name__=='__main__':
    main()
