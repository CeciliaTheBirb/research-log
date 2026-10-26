"""Gradient routing, conditioning alignment and exact resume of the conditional GAN."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from costream.video_gan import VideoGAN


class VideoGANTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        self.x = torch.randn(1, 4, 5, 8, 12).bfloat16()
        self.clip = SimpleNamespace(x0=self.x, anchor=self.x[:, :, :1],
                                    actions=torch.randn(17, 9), cid="training-clip")
        self.cfg = dict(weight=0.001, width=8, ramp_student_updates=2, r1_interval=1, r1_gamma=1)

    def make_gan(self):
        return VideoGAN(self.cfg, channels=4, action_dim=9, temporal_ratio=4,
                        device=torch.device("cpu"), seed=0)

    def test_gradient_isolation_and_lazy_r1(self):
        gan = self.make_gan()
        endpoint = self.x.float().clone().requires_grad_()
        loss, metrics = gan.generator_loss(self.clip, endpoint, step=0, student_updates=1)
        loss.backward()
        self.assertGreater(endpoint.grad[:, :, 1:].abs().sum().item(), 0)
        self.assertEqual(endpoint.grad[:, :, :1].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in gan.discriminator.parameters()))
        self.assertAlmostEqual(metrics["gan_weight"].item(), 0.0005)
        endpoint.grad = None
        dloss, dm = gan.discriminator_loss(self.clip, endpoint, step=1, fake_updates=1)
        dloss.backward()
        self.assertIsNone(endpoint.grad)
        self.assertGreater(dm["gan_r1"].item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gan.discriminator.parameters()))
        self.assertGreater(gan.step(sp_size=1), 0)

    def test_actions_anchor_and_temporal_order_affect_logits(self):
        gan = self.make_gan()
        video, anchor, actions, _ = gan._inputs(self.clip, self.x.float(), 0)
        d = gan.discriminator
        expected = d(video, anchor, actions)
        self.assertFalse(torch.allclose(expected, d(video, anchor, actions + 1)))
        self.assertFalse(torch.allclose(expected, d(video, anchor + 1, actions)))
        self.assertFalse(torch.allclose(expected, d(video.flip(2), anchor, actions)))
        self.clip.actions[0] += 1000  # null anchor action is not a transition
        self.assertTrue(torch.equal(actions, gan._inputs(self.clip, self.x.float(), 0)[2]))
        self.clip.actions = self.clip.actions[:10]
        with self.assertRaises(ValueError):
            gan._inputs(self.clip, self.x.float(), 0)

    def test_rng_and_zero_weight_do_not_change_base_gradient(self):
        before = torch.random.get_rng_state().clone()
        gan = self.make_gan()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        gan.weight = 0
        endpoint = self.x.float().clone().requires_grad_()
        gan_loss, _ = gan.generator_loss(self.clip, endpoint, step=0, student_updates=1)
        base = endpoint.square().mean()
        expected, = torch.autograd.grad(base, endpoint, retain_graph=True)
        actual, = torch.autograd.grad(base + gan_loss, endpoint)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_checkpoint_restores_exact_next_update_and_sp_scaling(self):
        original = self.make_gan()
        loss, _ = original.discriminator_loss(self.clip, self.x, step=1, fake_updates=1)
        loss.backward(); original.step(sp_size=1); original.zero_grad()
        with tempfile.TemporaryDirectory() as folder:
            original.save(Path(folder), 2)
            resumed = self.make_gan(); resumed.load(Path(folder), 2)
            for gan, sp in [(original, 1), (resumed, 4)]:
                loss, _ = gan.discriminator_loss(self.clip, self.x, step=3, fake_updates=2)
                (loss * sp).backward(); gan.step(sp_size=sp)
            for a, b in zip(original.discriminator.parameters(), resumed.discriminator.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
