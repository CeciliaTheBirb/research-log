"""CPU numerical audit of both architecture branches against the native vendor.

This is a reduced-width FP32 audit, not full-checkpoint / BF16 numerical parity.
The separate port-key audit checks every tensor of the full pinned checkpoints.
"""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import torch

from costream.model import packing
from costream.model.transformer import CosmosTransformer, TransformerConfig
from costream.parallel import SequenceLayout
from navworld_inference.vendored.cosmos3 import cosmos3_seq
from navworld_inference.vendored.cosmos3.transformer_cosmos3 import Cosmos3OmniTransformer


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    results = []
    for architecture, flags in {
        "Edge": dict(hidden_act="relu2", qk_norm_for_text=False, use_und_k_norm_for_gen=True),
        "Nano_Super": dict(hidden_act="silu", qk_norm_for_text=True, use_und_k_norm_for_gen=False),
    }.items():
        cfg = TransformerConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                rope_axes_dim=(4, 2, 2), vocab_size=64, latent_channel=4,
                                patch_latent_dim=16, action_dim=8, num_embodiment_domains=4,
                                two_time=False, **flags)
        kwargs = {k: v for k, v in asdict(cfg).items() if k in inspect.signature(Cosmos3OmniTransformer.__init__).parameters}
        kwargs.update(action_gen=True, enable_two_time=False, dtype="float32")
        torch.manual_seed(37)
        native = Cosmos3OmniTransformer(**kwargs).float().eval()
        port = CosmosTransformer(cfg).float().eval()
        source = native.state_dict()
        port.load_state_dict({k: source[k] for k in port.state_dict()}, strict=True)
        assert set(source) - set(port.state_dict()) == {"lm_head.weight"}
        for with_actions in (False, True):
            for seed in (0, 1):
                torch.manual_seed(seed)
                latent = torch.randn(1, 4, 5, 4, 6)
                ids = torch.randint(0, 64, (11,))
                act = torch.randn(16, 8) if with_actions else None
                if with_actions:
                    inputs = cosmos3_seq.build_forward_dynamics_inputs(native.config, ids, tuple(latent.shape), [0],
                                                                       650., 4, 20., act, 1, "cpu")
                    ref, _, _ = native(vision_tokens=[latent], action_tokens=[act], return_dict=False, **inputs.transformer_kwargs)
                else:
                    inputs, _ = cosmos3_seq.build_generation_inputs(native.config, ids, tuple(latent.shape), [0], 650., 4, 20., "cpu")
                    ref, _, _ = native(vision_tokens=[latent], return_dict=False, **inputs)
                packed = packing.pack_forward_dynamics(cfg, ids, tuple(latent.shape), condition_frames=(0,),
                                                        action_len=0 if act is None else len(act), fps=20., vae_temporal=4, device="cpu")
                times = torch.full((packed.loss_rows.numel(),), 650.)
                rows = port(packed, [latent], None if act is None else [act], 1, times,
                            layout=SequenceLayout(packed.seq_len, packed.und_len, None))
                actual = port.unpatchify(rows, packed.noisy_frames[-1], tuple(latent.shape))
                expected = ref[0]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
                error = (actual - expected).abs().max().item()
                result = dict(architecture=architecture, task="FD" if with_actions else "I2V", seed=seed,
                              max_abs_error=error, passed=True, shape=list(actual.shape))
                results.append(result)
                print(json.dumps(result), flush=True)
    paths = [Path(inspect.getfile(Cosmos3OmniTransformer)), Path(inspect.getfile(cosmos3_seq))]
    print(json.dumps(dict(complete=True, cases=len(results), precision="FP32", reduced_width=True,
                          reference_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})), flush=True)


if __name__ == "__main__":
    main()
