"""Pure-data design of the matched native-model controllability pilot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CASES = [
    dict(id="fd_stop_control", task="forward_dynamics", event="control", speed_mps=0., yaw_dps=0.),
    dict(id="fd_stop_person_ltr", task="forward_dynamics", event="person_ltr", speed_mps=0., yaw_dps=0.),
    dict(id="fd_stop_person_rtl", task="forward_dynamics", event="person_rtl", speed_mps=0., yaw_dps=0.),
    dict(id="fd_forward", task="forward_dynamics", event="control", speed_mps=.5, yaw_dps=0.),
    dict(id="fd_left", task="forward_dynamics", event="control", speed_mps=.5, yaw_dps=10.),
    dict(id="fd_right", task="forward_dynamics", event="control", speed_mps=.5, yaw_dps=-10.),
    dict(id="i2v_control", task="image_to_video", event="control"),
    dict(id="i2v_person_ltr", task="image_to_video", event="person_ltr"),
    dict(id="i2v_person_rtl", task="image_to_video", event="person_rtl"),
]

FORWARD_ENCOUNTER_CASES = [
    dict(id="fd_forward_control", task="forward_dynamics", event="control", speed_mps=.5, yaw_dps=0.),
    dict(id="fd_forward_oncoming_left", task="forward_dynamics", event="oncoming_left", speed_mps=.5, yaw_dps=0.),
    dict(id="fd_forward_oncoming_right", task="forward_dynamics", event="oncoming_right", speed_mps=.5, yaw_dps=0.),
]

EVENTS = {
    "control": "No additional pedestrian enters the foreground path in front of the robot. People already visible in the background remain consistent with the initial image.",
    "person_ltr": "One additional adult pedestrian enters from the left edge of the image, walks naturally from left to right across the path about three metres in front of the robot, and exits through the right edge. Left and right refer to the robot camera's image coordinates.",
    "person_rtl": "One additional adult pedestrian enters from the right edge of the image, walks naturally from right to left across the path about three metres in front of the robot, and exits through the left edge. Left and right refer to the robot camera's image coordinates.",
    "oncoming_left": "One adult pedestrian becomes visible about six metres ahead, facing the approaching robot and walking toward it at a natural walking pace. The pedestrian approaches from ahead, grows larger naturally, moves to the robot's LEFT side, passes beside the robot on its LEFT, and leaves through the lower-left edge by the end of the shot. LEFT means the left side of the robot camera image. The person approaches the camera rather than walking away; this is an oncoming encounter, not a distant sideways crossing.",
    "oncoming_right": "One adult pedestrian becomes visible about six metres ahead, facing the approaching robot and walking toward it at a natural walking pace. The pedestrian approaches from ahead, grows larger naturally, moves to the robot's RIGHT side, passes beside the robot on its RIGHT, and leaves through the lower-right edge by the end of the shot. RIGHT means the right side of the robot camera image. The person approaches the camera rather than walking away; this is an oncoming encounter, not a distant sideways crossing.",
}


def actions(case, num_frames=121, fps=20.):
    """120 body-frame SE(2) transitions, AV9 convention, no dummy anchor row.

    Positive yaw is a left turn. A stationary transition contains the identity
    rotation representation, rather than nine zeros (which is not a rotation).
    """
    speed, omega = case["speed_mps"], math.radians(case["yaw_dps"])
    angle = omega / fps
    forward = speed / fps if abs(omega) < 1e-12 else speed / omega * math.sin(angle)
    lateral = 0. if abs(omega) < 1e-12 else speed / omega * (1. - math.cos(angle))
    row = [-lateral, 0., forward, math.cos(angle), 0., math.sin(angle), 0., 1., 0.]
    return [list(row) for _ in range(num_frames - 1)]


def narrative(case, scene):
    if case["id"] in {c["id"] for c in FORWARD_ENCOUNTER_CASES}:
        # This suite tests the user's joint motion/event request. The numeric
        # trajectory is real conditioning, and agrees with the text in all cases.
        return (f"{scene} The robot moves straight forward steadily at 0.5 metres per second throughout the six-second shot, "
                "following the supplied numeric ego-motion actions. Its rigidly mounted low camera advances with realistic forward parallax, "
                "without turning, stopping, zooming, or changing mounting height. "
                f"{EVENTS[case['event']]} Keep the scene, lighting and background objects consistent with the initial image as the viewpoint advances. One continuous realistic shot.")
    # The FD camera trajectory is deliberately absent from text. Thus the
    # action comparison changes only numeric actions, with identical prompt IDs.
    camera = ("The camera is rigidly mounted on the mobile robot; its motion follows the supplied motion trajectory."
              if case["task"] == "forward_dynamics" else "The robot-mounted camera remains stationary throughout the shot.")
    return f"{scene} {camera} {EVENTS[case['event']]} Preserve the environment, viewpoint, lighting and objects visible in the initial image. One continuous realistic shot."


def i2v_caption(case, scene, num_frames=121, fps=20., height=480, width=832):
    description = narrative(case, scene)
    return json.dumps({
        "background_setting": scene,
        "cinematography": {"framing": "First-person view from a low mobile navigation robot.", "camera_motion": "Static camera."},
        "style_medium": "Live-action video",
        "actions": [{"time": "0:00-0:06", "description": EVENTS[case["event"]]}],
        "temporal_caption": description,
        "duration": "6s", "fps": fps, "resolution": {"H": height, "W": width}, "aspect_ratio": "16,9",
    }, ensure_ascii=False)


def expected_cases(spec):
    return sum(1 for a in spec["anchors"] for s in spec["seeds"] for c in spec["cases"]
               if c["task"] != "image_to_video" or (a == spec["anchors"][0] and s == spec["seeds"][0]))


def make_manifest(anchor_root: Path, suite="isolated"):
    if suite not in ("isolated", "forward_encounter"):
        raise ValueError(suite)
    anchors = [
        dict(id="rain", file="rain.png", source="NavWorldGym/test_assets/rain_obs.png",
             scene="A low navigation-robot camera observes the wet suburban intersection in the initial image, with a stone wall on the left, puddles, houses and parked cars."),
        dict(id="night0057", file="heldout-0057.png", source="held-out episode 0057, decoded reference frame 0",
             scene="A low navigation-robot camera observes the snowy urban sidewalk at night in the initial image, with warm decorative lights, buildings, parked vehicles and background pedestrians."),
    ]
    for anchor in anchors:
        anchor["image_sha256"] = hashlib.sha256((anchor_root / anchor["file"]).read_bytes()).hexdigest()
    spec = dict(schema_version=3, suite=suite, name="Native Super vs Nano vs Edge controllability pilot", date="2026-10-28",
                frames=121, fps=20., height=480, width=832, seeds=[2026, 2027], anchors=anchors,
                cases=CASES if suite == "isolated" else FORWARD_ENCOUNTER_CASES, steps=50, sampler="UniPC", precision="BF16",
                task_settings={"forward_dynamics": dict(guidance=1., shift=5., caption_format="action", action_domain="av"),
                               "image_to_video": dict(guidance=6., shift=10., caption_format="sft", action_domain="av")},
                native_weights=True, adapters=[], causal=False, noisy_kv_reuse=False,
                decoder="TinyVAE step38000 BF16 streaming",
                scope="Exploratory two-scene, two-seed controllability pilot, not a statistical model ranking. No population FVD is estimated from this small set. PSNR/SSIM lack paired real futures for these counterfactual events and trajectories.",
                action_interpretation="AV9 body-frame camera/ego transitions. These test navigation ego-motion, not robot arm or joint control.",
                image_repeat_frames_before_encoding=121, conditioned_latent_frames=[0],
                prompt_templates=dict(use_system_prompt=False, add_duration_template=False, add_resolution_template=False),
                input_policy="Shared image repeat-padded to 121 frames before VAE encoding for FD and I2V; only normalized latent frame zero is conditioned. Shared CPU FP32 initial noise and raw actions; model-specific native tokenizer. Future ground-truth frames are never inputs.")
    if suite == "forward_encounter":
        spec["name"] = "Native Super vs Nano vs Edge: forward motion with oncoming pedestrian passing left/right"
        spec["task_settings"] = {"forward_dynamics": spec["task_settings"]["forward_dynamics"]}
        spec["joint_control"] = dict(robot_speed_mps=.5, robot_yaw_dps=0., total_forward_metres=3.,
                                     pedestrian="approaches from ahead, then passes on robot-camera left/right",
                                     motion_text_agrees_with_numeric_actions=True)
    spec["records_per_model"] = expected_cases(spec)
    return spec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["isolated", "forward_encounter"], default="isolated")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(make_manifest(args.anchors, args.suite), indent=2) + "\n")
