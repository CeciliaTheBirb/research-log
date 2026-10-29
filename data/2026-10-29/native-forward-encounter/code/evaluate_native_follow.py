"""Matched unadapted Cosmos inference. Never loads a trained adapter or future video."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from costream.data import resize_center_crop
from costream.model.lora import LoRALinear
from costream.model.world_model import CosmosWorldModel, materialize
from costream.parallel import init_parallel
from costream.sampling import generate, write_mp4
from native_follow_cases import actions, i2v_caption, narrative


def node_path(value):
    p = str(value)
    if p.startswith("/p0/data/") and not Path("/p0/data/qianxun").exists():
        p = "/bigdata/" + p[len("/p0/data/"):]
    return Path(p)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def tensor_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def synchronize_ranks(world, device):
    # A one-rank HashStore NCCL barrier crashed before the first clip on gpu02.
    # There is nothing to synchronize in that case. Bind multi-rank barriers to
    # the already selected CUDA device rather than NCCL's rank-based guess.
    if world > 1:
        dist.barrier(device_ids=[device.index])


def vae_signature(root):
    cfg = json.loads((root / "vae/config.json").read_text())
    keys = ["latents_mean", "latents_std", "latent_channels", "z_dim", "scale_factor_temporal", "scale_factor_spatial"]
    return {k: cfg.get(k) for k in keys}


def ensure_prompt_fits(model, text):
    tokenizer = model.tokenizer
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template([dict(role="user", content=text)], tokenize=True,
                                           add_generation_prompt=True, return_dict=True)["input_ids"]
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) + 2 > model.max_text_tokens:
        raise ValueError(f"Prompt would be truncated: {len(ids) + 2} > {model.max_text_tokens}")
    return len(ids) + 2


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model", choices=["Edge", "Nano", "Super"], required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tinyvae", required=True)
    parser.add_argument("--shared-inputs", help="Reuse a separately verified canonical anchor/noise directory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(node_path(args.manifest).read_text())
    root, anchor_root, output = map(node_path, [args.model_root, args.anchors, args.output])
    marker = json.loads((root / "native-weights-verified.json").read_text())
    if marker["model"] != args.model or not marker["verified"] or marker["adapters"]:
        raise ValueError("Native checkpoint verification failed")
    torch.set_num_threads(2)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ps = init_parallel(world, device=device, fsdp_shard_size=world)
    model = CosmosWorldModel(str(root), two_time=False, action_domain="av", fps=spec["fps"],
                             max_text_tokens=2048, caption_format="action", causal=None, text_cache=world == 1)
    materialize(model, ps, adapters={}, lora_seed=0, activation_checkpointing=False, shard=world > 1)
    model.eval()
    parameter_count = sum(p.numel() for p in model.transformer.parameters())
    if any(isinstance(layer, LoRALinear) for layer in model.transformer.modules()) or model.cfg.two_time or model.causal is not None:
        raise AssertionError("An adapted/causal model entered the native comparison")
    shared = node_path(args.shared_inputs) if args.shared_inputs else output / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    signature = vae_signature(root)
    latent_shape = (1, 48, (spec["frames"] - 1) // 4 + 1, spec["height"] // 16, spec["width"] // 16)
    if ps.is_main:
        emit("preparing_canonical_inputs", model=args.model)
        for anchor in spec["anchors"]:
            image = anchor_root / anchor["file"]
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            if image_hash != anchor["image_sha256"]:
                raise ValueError(f"Anchor image hash mismatch: {image}")
            cached = shared / f"{anchor['id']}-repeat{spec['frames']}-anchor.safetensors"
            if not cached.exists():
                if args.model != "Edge":
                    raise RuntimeError("The Edge preparation must produce the shared anchor first")
                pixels = torch.from_numpy(np.array(Image.open(image).convert("RGB"))).permute(2, 0, 1)[None]
                pixels = resize_center_crop(pixels, spec["height"], spec["width"])
                pixels = (((pixels + 1.) * 127.5).round().clamp(0, 255) / 127.5 - 1.)
                # Official Diffusers 40e551a repeats the first image to the full
                # video length for BOTH FD and I2V, encodes it, then anchors only
                # latent frame zero. No actual future images enter this tensor.
                repeated = pixels.permute(1, 0, 2, 3)[None].to(device, dtype=model.dtype).expand(-1, -1, spec["frames"], -1, -1)
                raw = model.vae.encode(repeated).latent_dist.mode()
                mean = model.latents_mean.to(raw).view(1, -1, 1, 1, 1)
                inv_std = (1. / model.latents_std.to(raw)).view(1, -1, 1, 1, 1)
                encoded = ((raw - mean) * inv_std)[:, :, :1].contiguous()
                del raw, repeated
                if encoded.shape != (1, 48, 1, latent_shape[3], latent_shape[4]) or not torch.isfinite(encoded).all():
                    raise AssertionError(f"Invalid canonical anchor: {encoded.shape}")
                temporary = cached.with_suffix(".tmp")
                save_file({"anchor": encoded.cpu().contiguous()}, str(temporary),
                          metadata={"vae_signature": json.dumps(signature, sort_keys=True), "image_sha256": image_hash,
                                    "encoder_model": marker["repo_id"], "encoder_revision": marker["revision"],
                                    "image_repeat_frames": str(spec["frames"]), "conditioned_latent_frames": "0"})
                temporary.replace(cached)
            with safe_open(cached, framework="pt") as handle:
                metadata = handle.metadata()
                if (json.loads(metadata["vae_signature"]) != signature or metadata["image_sha256"] != image_hash
                        or metadata.get("image_repeat_frames") != str(spec["frames"])):
                    raise ValueError("Shared anchor VAE/image mismatch")
            emit("canonical_anchor_ready", anchor=anchor["id"], repeat_frames=spec["frames"])
        for seed in spec["seeds"]:
            cached = shared / f"noise-{seed}.safetensors"
            if not cached.exists():
                noise = torch.randn(latent_shape, dtype=torch.float32, generator=torch.Generator(device="cpu").manual_seed(seed))
                save_file({"noise": noise}, str(cached))
    synchronize_ranks(world, device)
    model.vae.to("cpu")
    torch.cuda.empty_cache()
    if args.prepare_only:
        dist.destroy_process_group()
        return
    from navworld_inference.tinyvae_checkpoint import load_tinyvae_decoder
    tiny = load_tinyvae_decoder(node_path(args.tinyvae), device, dtype=torch.bfloat16) if ps.is_main else None
    model_output = output / args.model.lower()
    if ps.is_main:
        save_json(model_output / "settings.json", {
            "model": args.model, "checkpoint": marker, "adapter_loaded": False, "lora_modules": 0,
            "generator_parameter_count": parameter_count,
            "backend": "CoStream Cosmos3 port, native one-time weights and bidirectional UniPC; not the distilled causal runtime",
            "native_one_time_bidirectional": True, "model_root": str(root), "gpu": torch.cuda.get_device_name(),
            "gpu_count": world, "sp": world, "fsdp_shards": world, "text_cache": world == 1,
            "job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__, "cuda": torch.version.cuda,
            "manifest_sha256": hashlib.sha256(node_path(args.manifest).read_bytes()).hexdigest(),
            "shared_inputs": str(shared), "suite": spec.get("suite", "isolated"),
            "tinyvae": str(node_path(args.tinyvae)), "tinyvae_dtype": "BF16", "tinyvae_parallel": False,
            "vae_signature": signature, "reference_metrics": "not applicable to counterfactual futures",
            "image_repeat_frames_before_encoding": spec["frames"], "conditioned_latent_frames": [0],
            "prompt_templates": {"use_system_prompt": False, "add_duration_template": False, "add_resolution_template": False,
                                 "forward_dynamics": "native action JSON", "image_to_video": "authored scene JSON"},
        })
    records = []
    for anchor_spec in spec["anchors"]:
        canonical_anchor = load_file(str(shared / f"{anchor_spec['id']}-repeat{spec['frames']}-anchor.safetensors"))["anchor"]
        dummy = torch.zeros(latent_shape, dtype=torch.bfloat16, device=device)
        dummy[:, :, :1] = canonical_anchor.to(device)
        for seed in spec["seeds"]:
            noise = load_file(str(shared / f"noise-{seed}.safetensors"))["noise"].to(device)
            for case in spec["cases"]:
                if args.limit and len(records) >= args.limit:
                    break
                # Auxiliary I2V probe uses only the rain anchor and first seed;
                # the FD joint-control matrix covers both anchors and both seeds.
                if case["task"] == "image_to_video" and (anchor_spec["id"] != spec["anchors"][0]["id"] or seed != spec["seeds"][0]):
                    continue
                folder = model_output / anchor_spec["id"] / f"seed-{seed}" / case["id"]
                report_path = folder / "result.json"
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    if report["manifest_sha256"] != hashlib.sha256(node_path(args.manifest).read_bytes()).hexdigest():
                        raise ValueError("Stale result belongs to a different suite")
                    records.append(report)
                    continue
                task = spec["task_settings"][case["task"]]
                model.caption_format = task["caption_format"]
                text = narrative(case, anchor_spec["scene"])
                camera_actions = None
                if case["task"] == "forward_dynamics":
                    camera_actions = torch.tensor(actions(case, spec["frames"], spec["fps"]), dtype=torch.float32, device=device)
                    negative = ""
                else:
                    text = i2v_caption(case, anchor_spec["scene"], spec["frames"], spec["fps"], spec["height"], spec["width"])
                    negative = json.dumps({"temporal_caption": "Blurry, distorted, temporally inconsistent video with deformed people, flickering objects, camera shake and sudden viewpoint changes."})
                clip = model.prepare_clip(latent=dummy, actions=camera_actions, caption=text, negative_prompt=negative,
                                          cid=f"{anchor_spec['id']}/{seed}/{case['id']}")
                # Native action CFG uses a raw negative user prompt, not another
                # action JSON. FD guidance=1 does not execute this branch.
                if case["task"] == "forward_dynamics":
                    clip.negative_ids = model._user_prompt_ids(negative).to(device)
                # For FD prompts inspect the actual structured wrapper, including
                # its media metadata, rather than only checking the raw narrative.
                original_max = model.max_text_tokens
                model.max_text_tokens = 65536
                full_ids = model.build_prompt_ids(text, spec["frames"], spec["height"], spec["width"])
                model.max_text_tokens = original_max
                if len(full_ids) != len(clip.input_ids) or not torch.equal(full_ids, clip.input_ids.cpu()):
                    raise ValueError("Positive prompt was truncated")
                if model.caption_format == "sft":
                    ensure_prompt_fits(model, negative)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                emit("native_follow_clip_started", model=args.model, anchor=anchor_spec["id"], seed=seed, case=case["id"])
                latents = generate(model, clip, adapter=None, mode="unipc", steps=spec["steps"],
                                   guidance=task["guidance"], shift=task["shift"], noise=noise, seed=seed)
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - start
                if not torch.isfinite(latents).all() or not torch.equal(latents[:, :, :1], clip.anchor.float()):
                    raise AssertionError("Nonfinite output or changed anchor")
                peak = torch.cuda.max_memory_allocated() / 2**30
                if ps.is_main:
                    started_decode = time.perf_counter()
                    pixels = tiny.decode(latents.permute(0, 2, 1, 3, 4).to(torch.bfloat16), start=True, end=True,
                                         parallel=False, to_neg_1_1=True)
                    if tuple(pixels.shape) != (1, spec["frames"], 3, spec["height"], spec["width"]) or not torch.isfinite(pixels).all():
                        raise AssertionError(f"Invalid TinyVAE output: {pixels.shape}")
                    torch.cuda.synchronize()
                    decode_seconds = time.perf_counter() - started_decode
                    frames = ((pixels[0].float().clamp(-1, 1) + 1.) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                    folder.mkdir(parents=True, exist_ok=True)
                    write_mp4(frames, folder / "generated.mp4", spec["fps"])
                    for frame in (0, 20, 40, 60, 80, 100, 120):
                        Image.fromarray(frames[frame]).save(folder / f"frame-{frame:03d}.jpg", quality=95)
                    save_file({"latents": latents.cpu().contiguous()}, str(folder / "generated.safetensors"))
                    report = dict(model=args.model, anchor=anchor_spec["id"], seed=seed, case=case,
                                  prompt=text, negative_prompt=negative, conditioning=task, prompt_tokens=len(clip.input_ids), prompt_truncated=False,
                                  prompt_ids_sha256=hashlib.sha256(clip.input_ids.cpu().numpy().tobytes()).hexdigest(),
                                  raw_actions_sha256=None if camera_actions is None else tensor_hash(camera_actions),
                                  raw_actions_shape=None if camera_actions is None else list(camera_actions.shape),
                                  anchor_sha256=tensor_hash(canonical_anchor), initial_noise_sha256=tensor_hash(noise),
                                  latent_sha256=tensor_hash(latents), rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                                  generation_seconds=generation_seconds, tiny_decode_seconds=decode_seconds,
                                  generation_and_decode_fps=(spec["frames"]-1)/(generation_seconds+decode_seconds),
                                  peak_generation_allocated_gib=peak, gpu=torch.cuda.get_device_name(), gpu_count=world,
                                  timing_scope="Excludes loading, anchor encoding, file writes. Different model allocations; not a matched-hardware speed benchmark.",
                                  manifest_sha256=hashlib.sha256(node_path(args.manifest).read_bytes()).hexdigest())
                    save_json(report_path, report)
                    emit("native_follow_clip_complete", model=args.model, anchor=anchor_spec["id"], seed=seed, case=case["id"], seconds=generation_seconds)
                else:
                    report = {"case": case}
                records.append(report)
                synchronize_ranks(world, device)
    if ps.is_main:
        save_json(model_output / ("smoke-complete.json" if args.limit else "complete.json"),
                  dict(model=args.model, complete=True, records=len(records), limited=bool(args.limit),
                       cases=[dict(anchor=r["anchor"], seed=r["seed"], case=r["case"]["id"]) for r in records]))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
