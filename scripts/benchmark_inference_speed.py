"""Ablate single-device execution and text caching on identical checkpoints/noise."""
import argparse
import copy
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file

from costream.config import load_config
from costream.data import ClipDataset
from costream.parallel import init_parallel
from costream.rollout import CausalPolicy
from costream.sampling import generate, write_mp4
from costream.train import build_model
from evaluate_stage3 import node_path, prepare_evaluation_clip, restore, save_json, tiny_decode, uint8_frames
from inference import load_model, read_model_config
from navworld_inference.tinyvae_checkpoint import load_tinyvae_decoder


def rgb_difference(a, b):
    total = changed = maximum = 0
    for start in range(0, len(a), 8):
        delta = np.abs(np.subtract(a[start:start+8], b[start:start+8], dtype=np.int16))
        total += int(delta.sum())
        changed += int(np.count_nonzero(delta))
        maximum = max(maximum, int(delta.max()))
    return {'rgb_equal': changed == 0, 'rgb_mae_0_255': total/a.size,
            'rgb_changed_fraction': changed/a.size, 'rgb_max_difference': maximum}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--indices', type=int, nargs='+', default=[0, 57, 114])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--optimized-only', action='store_true', help='Only benchmark deployed single-device, text-cached students.')
    a = p.parse_args()
    m = json.loads(node_path(a.manifest).read_text())
    root = node_path(a.output)
    if not root.is_absolute():
        raise ValueError('--output must be an absolute persistent directory, outside job scratch')
    root.mkdir(parents=True, exist_ok=True)
    assert m['tiny_parallel'] is False
    torch.cuda.set_device(0); torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    base_overrides = ['parallel.sp_size=1', 'parallel.fsdp_shard_size=1', 'data.stage=false']
    cfg = load_config('configs/stage2_edge_navworld_422_clean.yaml', base_overrides)
    dataset = ClipDataset(cfg.data.root, 'val', index=cfg.data.val_index, caption_mode='none')
    items = {i: dataset[i] for i in a.indices}
    noise_manifest = json.loads((node_path(m['initial_noise_dir'])/'manifest.json').read_text())
    noises = {}
    for i in a.indices:
        seed = m['seed']+i
        noise = load_file(str(node_path(m['initial_noise_dir'])/f'seed{seed}.safetensors'))['noise']
        assert hashlib.sha256(noise.numpy().tobytes()).hexdigest() == noise_manifest['noise_sha256'][str(seed)]
        noises[i] = noise.to(device)
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'job_id': os.environ.get('SLURM_JOB_ID'), 'indices': a.indices,
              'warmup_per_clip': 1, 'timed_repeats_per_clip': a.repeats,
              'protocol': '120 new RGB frames / synchronized generation plus TinyVAE BF16 streaming decode; '
                          'same GPU, checkpoints, prompts, action layout and canonical noise. '
                          'Every video rebuilds its own text cache; loading, anchor encoding, CPU conversion, MP4 and quality validation excluded.',
              'manifest': m, 'records': [], 'validation': [], 'runtimes': {}}
    baseline = {}
    tiny = load_tinyvae_decoder(node_path(m['tinyvae']), device, dtype=torch.bfloat16)

    def measure(variant, runtime, index, sampler, stats, cache_check=None):
        snapshot = None
        for repeat in range(-1, a.repeats):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            latent = sampler()
            torch.cuda.synchronize(); midpoint = time.perf_counter()
            pixels = tiny_decode(tiny, latent, False)
            torch.cuda.synchronize(); end = time.perf_counter()
            row = {'variant': variant, 'runtime': runtime, 'index': index, 'repeat': repeat,
                   'rollout_s': midpoint-start, 'decode_s': end-midpoint, 'combined_s': end-start,
                   'combined_fps': 120/(end-start), 'calls': copy.deepcopy(stats()),
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
            assert torch.isfinite(latent).all() and torch.isfinite(pixels).all()
            if repeat == 0:
                snapshot = (latent.cpu(), uint8_frames(pixels))
                if cache_check: row['text_cache'] = cache_check()
            del latent, pixels
            if repeat >= 0:
                result['records'].append(row)
                print(json.dumps(row), flush=True)
        if runtime == 'fsdp_no_text_cache':
            baseline[variant, index] = snapshot
        elif runtime != 'original_native' and not a.optimized_only:
            previous_latent, previous_rgb = baseline[variant, index]
            latent, rgb = snapshot
            delta = latent.float()-previous_latent.float()
            comparison = {'variant': variant, 'runtime': runtime, 'index': index,
                          'latent_equal': torch.equal(latent, previous_latent),
                          'latent_max_abs': float(delta.abs().max()),
                          'latent_rmse': float(delta.square().mean().sqrt()),
                          'latent_relative_l2': float(delta.norm()/previous_latent.float().norm().clamp_min(1e-12)),
                          **rgb_difference(rgb, previous_rgb)}
            result['validation'].append(comparison)
            print(json.dumps({'validation': comparison}), flush=True)
        if runtime in ['fsdp_no_text_cache', 'single_device_text_cache']:
            folder = root/runtime/variant/f'{index:04d}'; folder.mkdir(parents=True, exist_ok=True)
            write_mp4(snapshot[1], folder/'generated.mp4', 20)
        save_json(root/'progress.json', result)

    checkpoint = node_path(m['native_checkpoint'])
    params, filters = read_model_config(checkpoint/'inference_config.yaml', node_path(cfg.model.root))
    native, _ = load_model(params, checkpoint, filters, device,
                           checkpoint/'action_weights.safetensors', checkpoint/'time_weights.safetensors')
    native.vae.to('cpu')
    assert native.cache_text, 'Old native student must use its default text cache'
    assert not any('FSDP' in type(module).__name__ for module in native.modules())
    result['runtimes']['original_native'] = {'single_device': True, 'fsdp': False, 'text_cache': bool(native.cache_text),
                                             'cache_reset': 'new per-layer generation cache constructed inside each generate request'}
    for i in a.indices:
        anchor = items[i]['latent'][None, :, :1].to(device=device, dtype=torch.bfloat16)
        raw = items[i]['actions']; assert raw.shape == (121,9) and torch.count_nonzero(raw[0]) == 0
        actions = raw[1:].to(device)
        def native_sample():
            calls = []
            def block_noise(shape, **kwargs):
                assert tuple(shape) == (1,48,3,30,52) and len(calls) < 10
                start = 1+3*len(calls); calls.append(start)
                return noises[i][:,:,start:start+3].to(device=kwargs['device'], dtype=kwargs['dtype']).clone()
            with patch.object(torch, 'randn', block_noise):
                latent = native.generate(prompt='You are an autonomous vehicle planning system.',
                                         negative_prompt='', num_frames=121, height=480, width=832,
                                         num_inference_steps=4, guidance_scale=1., flow_shift=5.,
                                         first_frame_latent=anchor, camera_actions=actions,
                                         seed=m['seed']+i, decode_output=False)
            assert len(calls) == 10
            return latent
        measure('old4444_native', 'original_native', i, native_sample, lambda: native.generation_metadata)
    del native, anchor, actions
    gc.collect(); torch.cuda.empty_cache()

    ps = init_parallel(1, device=device, fsdp_shard_size=1)
    runtime_modes = [('single_device_text_cache', True, True)] if a.optimized_only else [('fsdp_no_text_cache', False, False),
                                       ('single_device_no_text_cache', True, False),
                                       ('single_device_text_cache', True, True)]
    for runtime, plain, text_cache in runtime_modes:
        cfg = load_config('configs/stage2_edge_navworld_422_clean.yaml', base_overrides+[
            f'inference.single_device={str(plain).lower()}', f'inference.text_cache={str(text_cache).lower()}'])
        model = build_model(cfg, ps, inference=True).eval(); model.vae.to('cpu')
        result['runtimes'][runtime] = model.inference_runtime
        for variant in m['variants'][1:]:
            restore(model, variant, m['initializer'])
            for i in a.indices:
                clip = prepare_evaluation_clip(model, dataset, i, cfg, variant, device)
                stats = {}
                def student_sample():
                    clip.text_cache.clear()
                    return generate(model, clip, adapter='student', mode='causal', steps=4,
                                    guidance=1., shift=5., seed=m['seed']+i,
                                    policy=CausalPolicy.from_config(variant['policy']), noise=noises[i], stats=stats)
                def check_cache():
                    cached = clip.text_cache.get(id(clip.input_ids), {})
                    count = len(cached.get('kv', []))
                    if text_cache: assert count == len(model.transformer.layers), 'text cache did not persist'
                    return {'enabled': model.text_cache, 'layer_count': count, 'tokens': len(clip.input_ids)}
                measure(variant['name'], runtime, i, student_sample, lambda: stats, check_cache)
                del clip
        del model
        gc.collect(); torch.cuda.empty_cache()
    result['summary'] = {}
    for variant, runtime in sorted({(r['variant'],r['runtime']) for r in result['records']}):
        rows = [r for r in result['records'] if (r['variant'],r['runtime']) == (variant,runtime)]
        per_clip = {}
        for i in a.indices:
            selected = [r for r in rows if r['index'] == i]
            per_clip[str(i)] = {k: float(np.median([r[k] for r in selected])) for k in ['rollout_s','decode_s','combined_s']}
            per_clip[str(i)]['combined_fps'] = 120/per_clip[str(i)]['combined_s']
        combined = sum(v['combined_s'] for v in per_clip.values())
        result['summary'][variant+'__'+runtime] = {'variant': variant, 'runtime': runtime,
            'combined_fps': 120*len(a.indices)/combined, 'per_clip_median': per_clip}
    result['complete'] = True
    save_json(root/'benchmark.json', result)
    print(json.dumps({'complete': True, 'summary': result['summary']}), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
