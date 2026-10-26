"""Small action-conditional video GAN in the normalized Cosmos latent space.

An independent, replicated FP32 discriminator is trained alongside fake-score
updates. It is never part of the inference model. This is a conditional GAN
ablation, not the pretrained score-feature discriminator from the DMD2 paper.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


class ActionVideoDiscriminator(nn.Module):
    """Temporal PatchGAN with anchor, motion differences and per-time action projection."""

    def __init__(self, channels: int, action_dim: int, temporal_ratio: int = 4, width: int = 32):
        super().__init__()
        self.action_dim, self.temporal_ratio = action_dim, temporal_ratio
        blocks = []
        incoming = 3 * channels + 1
        for outgoing, dilation in [(width, 1), (2 * width, 2), (4 * width, 4)]:
            blocks += [nn.Conv3d(incoming, outgoing, 3, stride=(1, 2, 2),
                                 padding=(dilation, 1, 1), dilation=(dilation, 1, 1)),
                       nn.GroupNorm(8, outgoing), nn.LeakyReLU(0.2)]
            incoming = outgoing
        blocks += [nn.Conv3d(incoming, incoming, (3, 1, 1), padding=(8, 0, 0), dilation=(8, 1, 1)),
                   nn.LeakyReLU(0.2)]
        self.video = nn.Sequential(*blocks)
        self.unconditional = nn.Conv3d(incoming, 1, 1)
        self.action = nn.Sequential(nn.Linear(action_dim * temporal_ratio, incoming), nn.SiLU(),
                                    nn.Linear(incoming, incoming))

    def forward(self, future: torch.Tensor, anchor: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
        # future: [1,C,T-1,H,W]. Each temporal cell conditions on its own four RGB transitions.
        b, _, t, h, w = future.shape
        if transitions.shape != (t * self.temporal_ratio, self.action_dim):
            raise ValueError(f"unaligned GAN actions: {transitions.shape}, latent time {t}")
        first = anchor.detach().expand(b, -1, t, -1, -1)
        previous = torch.cat([anchor.detach(), future[:, :, :-1]], dim=2)
        clock = torch.linspace(0, 1, t, device=future.device).view(1, 1, t, 1, 1).expand(b, 1, t, h, w)
        features = self.video(torch.cat([future, first, future - previous, clock], dim=1).float())
        actions = transitions.detach().float().asinh().reshape(t, -1)
        condition = self.action(actions).transpose(0, 1).unsqueeze(0)
        projection = (features * condition[:, :, :, None, None]).sum(1, keepdim=True) / math.sqrt(features.shape[1])
        # Retain spatiotemporal patch logits; logistic losses are applied before averaging.
        return self.unconditional(features) + projection


class VideoGAN:
    def __init__(self, cfg: dict, *, channels: int, action_dim: int, temporal_ratio: int,
                 device: torch.device, seed: int, action_layout: str = "anchor_padded"):
        self.cfg = dict(cfg)
        self.device, self.seed, self.action_layout = device, int(seed), action_layout
        self.temporal_ratio, self.action_dim = int(temporal_ratio), int(action_dim)
        if action_layout not in ("anchor_padded", "transitions"):
            raise ValueError(action_layout)
        self.weight = float(cfg.get("weight", 0.001))
        self.ramp_student_updates = int(cfg.get("ramp_student_updates", 50))
        self.instance_noise = float(cfg.get("instance_noise", 0.01))
        self.r1_gamma, self.r1_interval = float(cfg.get("r1_gamma", 1)), int(cfg.get("r1_interval", 16))
        if self.weight < 0 or self.ramp_student_updates < 0 or self.r1_interval < 1:
            raise ValueError("invalid GAN weight/ramp/R1 interval")
        # Construct on CPU inside an RNG fork: adding GAN cannot change model/data RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.seed + 93017)
            self.discriminator = ActionVideoDiscriminator(channels, action_dim, temporal_ratio,
                                                          int(cfg.get("width", 32))).to(device)
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=float(cfg.get("lr", 1e-4)),
                                           betas=(0.0, 0.99), weight_decay=0.0)

    def _inputs(self, clip, endpoint: torch.Tensor, step: int):
        if clip.actions is None:
            raise ValueError("conditional video GAN requires actions")
        future = endpoint[:, :, 1:].to(clip.x0.dtype).float()
        anchor = clip.anchor.detach().float()
        count = int(future.shape[2]) * self.temporal_ratio
        offset = int(self.action_layout == "anchor_padded")
        transitions = clip.actions[offset:offset + count].detach().float()
        if transitions.shape != (count, self.action_dim):
            raise ValueError(f"expected {count} action transitions, got {transitions.shape}")
        # Quantize both real and generated latents identically, then add equal-variance
        # instance noise. This avoids a BF16-real/FP32-generated precision shortcut.
        digest = hashlib.blake2b(f"{self.seed}:{step}:{clip.cid}".encode(), digest_size=8).digest()
        rng = torch.Generator(device=self.device).manual_seed(int.from_bytes(digest, "little"))
        noise = torch.randn(future.shape, device=self.device, generator=rng, dtype=torch.float32)
        noise = noise * self.instance_noise
        return future + noise, anchor, transitions, noise

    def generator_loss(self, clip, endpoint: torch.Tensor, *, step: int, student_updates: int):
        future, anchor, actions, _ = self._inputs(clip, endpoint, step)
        frozen = {name: p.detach() for name, p in self.discriminator.named_parameters()}
        logits = functional_call(self.discriminator, frozen, (future, anchor, actions))
        raw = F.softplus(-logits).mean()
        ramp = min(student_updates / self.ramp_student_updates, 1.0) if self.ramp_student_updates else 1.0
        weight = self.weight * ramp
        return raw * weight, {"gan_g_loss": raw.detach(), "gan_weight": raw.new_tensor(weight),
                              "gan_g_weighted": (raw * weight).detach(), "gan_g_fake_logit": logits.detach().mean()}

    def discriminator_loss(self, clip, endpoint: torch.Tensor, *, step: int, fake_updates: int):
        future, anchor, actions, noise = self._inputs(clip, endpoint.detach(), step)
        real = clip.x0[:, :, 1:].detach().to(clip.x0.dtype).float() + noise
        apply_r1 = self.r1_gamma > 0 and fake_updates % self.r1_interval == 0
        real.requires_grad_(apply_r1)
        real_logits = self.discriminator(real, anchor, actions)
        fake_logits = self.discriminator(future.detach(), anchor, actions)
        raw = (F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()) * 0.5
        r1 = raw.new_zeros(())
        if apply_r1:
            gradient = torch.autograd.grad(real_logits.mean(), real, create_graph=True)[0]
            r1 = gradient.float().square().flatten(1).sum(1).mean()
        loss = raw + 0.5 * self.r1_gamma * self.r1_interval * r1
        return loss, {"gan_d_loss": raw.detach(), "gan_r1": r1.detach(),
                      "gan_d_real_logit": real_logits.detach().mean(), "gan_d_fake_logit": fake_logits.detach().mean()}

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, *, sp_size: int) -> float:
        params = list(self.discriminator.parameters())
        # The trainer scales loss by SP for the sharded transformer; this separate
        # replicated CNN needs that factor removed, then a mean over all DP/SP ranks.
        for param in params:
            if param.grad is None:
                raise RuntimeError("GAN parameter missing a discriminator gradient")
            param.grad.div_(sp_size)
            if dist.is_initialized():
                dist.all_reduce(param.grad)
                param.grad.div_(dist.get_world_size())
        norm = torch.nn.utils.clip_grad_norm_(params, float(self.cfg.get("grad_clip", 1)))
        finite = torch.isfinite(norm).to(torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("nonfinite conditional video GAN gradients")
        self.optimizer.step()
        return float(norm)

    def save(self, path: Path, step: int):
        if not dist.is_initialized() or dist.get_rank() == 0:
            torch.save({"step": step, "config": self.cfg, "model": self.discriminator.state_dict(),
                        "optimizer": self.optimizer.state_dict()}, Path(path) / "video_gan.pt")
        if dist.is_initialized():
            dist.barrier()

    def load(self, path: Path, step: int):
        data = torch.load(Path(path) / "video_gan.pt", map_location=self.device, weights_only=True)
        if data["step"] != step or data["config"] != self.cfg:
            raise ValueError("GAN checkpoint step/config mismatch")
        self.discriminator.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
