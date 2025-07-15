# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/sam2/blob/main/sam2/benchmark.py

import os
import time

import numpy as np
import torch

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    raise RuntimeError("No CUDA or MPS device found")

# Config and checkpoint
# model_cfg = "configs/efficienttam/efficienttam_s.yaml"
# model_cfg = "configs/efficienttam/efficienttam_s_1.yaml"
# model_cfg = "configs/efficienttam/efficienttam_s_2.yaml"
model_cfg = "configs/efficienttam/efficienttam_s_512x512.yaml"
# model_cfg = "configs/efficienttam/efficienttam_ti.yaml"
# model_cfg = "configs/efficienttam/efficienttam_ti_1.yaml"
# model_cfg = "configs/efficienttam/efficienttam_ti_2.yaml"
# model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"
efficienttam_checkpoint = None

# Build video predictor with vos_optimized=True setting
predictor = build_efficienttam_video_predictor(
    model_cfg, efficienttam_checkpoint, device=device, vos_optimized=True
)

model_total_params = sum(p.numel() for p in predictor.parameters())
print("Model Size: ", model_total_params)

# Initialize with video
video_dir = "notebooks/videos/bedroom"
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir)


# Number of runs, warmup etc
warm_up, runs = 5, 25
verbose = True
num_frames = len(frame_names)
total, count = 0, 0
torch.cuda.empty_cache()

# We will select an object with a click.
# See video_predictor_example.ipynb for more detailed explanation
ann_frame_idx, ann_obj_id = 0, 1
# Add a positive click at (x, y) = (210, 350)
# For labels, `1` means positive click
points = np.array([[210, 350]], dtype=np.float32)
labels = np.array([1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Warmup and then average FPS over several runs
with torch.inference_mode():
    for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
        start = time.time()
        # Start tracking
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in predictor.propagate_in_video(inference_state):
            pass

        end = time.time()
        total += end - start
        count += 1
        if i == warm_up - 1:
            print("Warmup FPS: ", count * num_frames / total)
            total = 0
            count = 0

print("FPS: ", count * num_frames / total)
