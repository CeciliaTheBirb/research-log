"""Evaluate the pinned HF Hybrid Forcing export with its matching native runtime.

Uses the existing validation inputs, raw RGB references, canonical noise and
metrics. No future latent or future caption is passed to the model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file


def mapped(value):
    value = str(value)
    if value.startswith('/p0/data/') and not Path('/p0/data/qianxun').exists():
        value = '/bigdata/' + value[len('/p0/data/'):]
    return Path(value)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['evaluate', 'benchmark', 'rain'], required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=172)
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(mapped(args.manifest).read_text())
    runtime = mapped(manifest['runtime'])
    sys.path.insert(0, str(runtime))
    from inference import load_model, read_model_config, read_generation_defaults
    from navworld_inference.tinyvae_checkpoint import load_tinyvae_decoder
    from evaluate_stage3 import Quality, reference_frames, save_json, tiny_decode, uint8_frames
    from costream.config import load_config
    from costream.data import ClipDataset
    from costream.sampling import write_mp4

    output = mapped(args.output)
    if not output.is_absolute():
        raise ValueError('Output must be an absolute persistent path')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = mapped(manifest['hybrid_checkpoint'])
    published = json.loads((checkpoint / 'manifest.json').read_text())
    for name, entry in published['files'].items():
        payload = checkpoint / name
        assert payload.stat().st_size == entry['bytes'], name
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == entry['sha256'], name
    assert Path(sys.modules['inference'].__file__).resolve() == (runtime/'inference.py').resolve()
    cfg = load_config('configs/stage2_edge_navworld_422_clean.yaml', ['data.stage=false'])
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    params, filters = read_model_config(checkpoint/'inference_config.yaml', mapped(cfg.model.root))
    model, loaded = load_model(params, checkpoint, filters, device,
                              checkpoint/'action_weights.safetensors', checkpoint/'time_weights.safetensors',
                              checkpoint/'history_weights.safetensors')
    model.vae.to('cpu')
    torch.cuda.empty_cache()
    assert loaded['adapter_tensors'] == 344 and loaded['history_tensors'] == 28
    assert model.cache_text and not any('FSDP' in type(module).__name__ for module in model.modules())
    rotary = model._raw_transformer().rotary_emb
    assert rotary.inv_freq.dtype == torch.float32, 'Preserve official training RoPE precision'
    tiny = load_tinyvae_decoder(mapped(manifest['tinyvae']), device, dtype=torch.bfloat16)
    defaults = read_generation_defaults(checkpoint/'inference_config.yaml')
    assert defaults == {'attention_sink_size': 1, 'attention_history_cutoff': 2,
                        'inference_step_schedule': [4,3,2,2,2,2,2,2,2,2]}
    variant = 'hybrid2400_nocaption'
    settings = {'variant': variant, 'hf_source': manifest['hf_source'], 'runtime_commit': manifest['runtime_commit'],
                'loaded': loaded, 'gpu': torch.cuda.get_device_name(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'mode': args.mode,
                'single_device': True, 'text_cache': bool(model.cache_text), 'rope_inv_freq_dtype': str(rotary.inv_freq.dtype),
                'generation': defaults, 'conditioning': {'caption_format': 'sft', 'prompt': '', 'max_text_tokens': 2048},
                'tinyvae': manifest['tinyvae'], 'tiny_parallel': False, 'future_latents_passed': False,
                'index_slice': [args.start, args.stop], 'input_scope': 'Anchor plus recorded actions; no future captions'}
    save_json(output/f'settings-{args.mode}-{args.start}-{args.stop}.json', settings)
    print(json.dumps({'event':'hybrid_loaded', **settings}), flush=True)

    def generate(anchor, actions, noise, seed):
        draws = []
        def block_noise(shape, **kwargs):
            assert tuple(shape) == (1,48,3,30,52) and len(draws) < 10
            start = 1 + 3*len(draws)
            draws.append(start)
            return noise[:,:,start:start+3].to(device=kwargs['device'], dtype=kwargs['dtype']).clone()
        with patch.object(torch, 'randn', block_noise):
            latent = model.generate(prompt='', negative_prompt='', num_frames=121, height=480, width=832,
                                    num_inference_steps=4, guidance_scale=1., flow_shift=5.,
                                    first_frame_latent=anchor, camera_actions=actions, seed=seed,
                                    decode_output=False, **defaults)
        calls = copy.deepcopy(model.generation_metadata)
        assert len(draws) == 10, f'Expected ten noise blocks, got {len(draws)}'
        assert list(calls['per_generated_block_nfe']) == defaults['inference_step_schedule'], calls
        assert calls['denoise_call_count'] == 23 and calls['cache_commit_count'] == 11
        assert calls['history_memory']['kind'] == 'linear_attention_v1'
        assert calls['attention_window'] == {'sink_size':1, 'history_cutoff':2}
        return latent, calls

    def sample(anchor, actions, noise, seed):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        latent, calls = generate(anchor, actions, noise, seed)
        torch.cuda.synchronize(); midpoint = time.perf_counter()
        pixels = tiny_decode(tiny, latent, False)
        torch.cuda.synchronize(); end = time.perf_counter()
        assert torch.isfinite(latent).all()
        timing = {'rollout_s':midpoint-start, 'decode_s':end-midpoint, 'combined_s':end-start,
                  'combined_fps':120/(end-start), 'rollout_fps':120/(midpoint-start), 'calls':calls,
                  'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        return uint8_frames(pixels), timing

    if args.mode == 'rain':
        records = []
        for case, value in manifest['rain_inputs'].items():
            source = mapped(value)
            tensors = load_file(str(source/'rain/latents.safetensors'))
            actions = torch.from_numpy(np.load(source/'rain/actions.npy')).to(device)
            assert actions.shape == (120,9)
            noise = tensors['noise'].to(device)
            frames, timing = sample(tensors['anchor'].to(device), actions, noise, manifest['seed'])
            folder = output/'rain'/case
            folder.mkdir(parents=True, exist_ok=True)
            write_mp4(frames, folder/'generated.mp4', 20)
            np.savez_compressed(folder/'contact_frames.npz', pred=frames[[0,20,60,120]])
            records.append({'case':case, 'source':str(source), 'timing':timing,
                            'actions_sha256':hashlib.sha256(actions.cpu().numpy().tobytes()).hexdigest(),
                            'noise_sha256':hashlib.sha256(tensors['noise'].numpy().tobytes()).hexdigest(),
                            'rgb_sha256':hashlib.sha256(frames.tobytes()).hexdigest()})
        save_json(output/'rain/audit.json', {'complete':True, 'settings':settings, 'records':records})
        return

    dataset = ClipDataset(cfg.data.root, 'val', index=cfg.data.val_index, caption_mode='none')
    assert len(dataset) == 172
    noise_root = mapped(manifest['initial_noise_dir'])
    noise_hashes = json.loads((noise_root/'manifest.json').read_text())['noise_sha256']
    quality = Quality(mapped(manifest['assets']), device) if args.mode == 'evaluate' else None
    indices = args.indices if args.indices is not None else list(range(args.start, args.stop))
    benchmark = {'settings':settings, 'records':[], 'indices':indices, 'warmup_per_clip':1,
                 'timed_repeats_per_clip':args.repeats, 'protocol':manifest['fps_protocol']}
    for i in indices:
        folder = output/variant/f'{i:04d}'
        if args.mode == 'evaluate' and (folder/'metrics.json').exists():
            continue
        item, record = dataset[i], dataset.records[i]
        seed = manifest['seed'] + i
        noise = load_file(str(noise_root/f'seed{seed}.safetensors'))['noise']
        digest = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
        assert digest == noise_hashes[str(seed)]
        noise = noise.to(device)
        anchor = item['latent'][None,:,:1].to(device=device,dtype=torch.bfloat16)
        raw = item['actions']
        assert raw.shape == (121,9) and torch.count_nonzero(raw[0]) == 0
        actions = raw[1:].to(device)
        prompt_ids = model.build_prompt_ids('',121,480,832)
        started = time.perf_counter()
        for repeat in (range(-1,args.repeats) if args.mode == 'benchmark' else [0]):
            frames, timing = sample(anchor, actions, noise, seed)
            timing['initial_noise_sha256'] = digest
            timing['rgb_sha256'] = hashlib.sha256(frames.tobytes()).hexdigest()
            if args.mode == 'benchmark':
                if repeat >= 0:
                    row = {'variant':variant,'index':i,'repeat':repeat, **timing}
                    benchmark['records'].append(row)
                    print(json.dumps(row),flush=True)
                    save_json(output/'benchmark-progress.json',benchmark)
                if repeat == 0:
                    folder.mkdir(parents=True,exist_ok=True)
                    write_mp4(frames,folder/'generated.mp4',20)
            else:
                real = reference_frames(record,mapped(manifest['reference_root']))
                paired = quality.paired(frames,real)
                features, real_features = quality.features(frames), quality.features(real)
                folder.mkdir(parents=True,exist_ok=True)
                np.savez(folder/'features.npz', **{'pred_'+k:v for k,v in features.items()},
                         **{'real_'+k:v for k,v in real_features.items()})
                write_mp4(frames,folder/'generated.mp4',20)
                np.savez_compressed(folder/'contact_frames.npz', indices=np.array([1,20,60,120]),
                                    pred=frames[[1,20,60,120]],real=real[[1,20,60,120]])
                save_json(folder/'metrics.json', {'index':i,'cid':item['cid'],'seed':seed,'variant':variant,
                          'actions':{'rows':120,'null_anchor_removed':True,
                                     'sha256':hashlib.sha256(actions.cpu().numpy().tobytes()).hexdigest()},
                          'prompt_tokens':len(prompt_ids),
                          'prompt_ids_sha256':hashlib.sha256(prompt_ids.numpy().tobytes()).hexdigest(),
                          'quality':paired,'timing':timing,'elapsed_s':time.perf_counter()-started})
                print(json.dumps({'event':'hybrid_clip_done','index':i,'psnr':paired['psnr_db'],
                                  'lpips':paired['lpips_alex'],'elapsed_s':time.perf_counter()-started}),flush=True)
    if args.mode == 'benchmark':
        medians = {str(i):{k:float(np.median([r[k] for r in benchmark['records'] if r['index']==i]))
                          for k in ['rollout_s','decode_s','combined_s']} for i in indices}
        benchmark['summary'] = {'variant':variant,'per_clip_median':medians,
                                'combined_fps':120*len(indices)/sum(r['combined_s'] for r in medians.values())}
        benchmark['complete'] = True
        save_json(output/'benchmark.json',benchmark)
        print(json.dumps({'complete':True,'summary':benchmark['summary']}),flush=True)


if __name__ == '__main__':
    main()
