import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np
import matplotlib.pyplot as plt
from decord import VideoReader
from decord import cpu, gpu
import PIL.Image as Image
import os

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

from torch.cuda.amp import autocast

def get_safe_autocast():
    if not torch.cuda.is_available():
        return autocast(enabled=False)

    major, minor = torch.cuda.get_device_capability()
    # Ampere (8.0+) supporta BF16 e Tensor Cores
    if major >= 8:
        return autocast(dtype=torch.bfloat16)
    # Volta/Turing (7.x) può supportare FP16
    elif major >= 7:
        return autocast(dtype=torch.float16)
    else:
        # GPU vecchie (Maxwell, Pascal): no mixed precision
        return autocast(enabled=False)

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.autocast("cuda", dtype=torch.float32).__enter__()  # fallback sicuro

    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
video = "/home/droghini.d@ICS.LOCAL/Downloads/split_stream/50_frames/mediterranea__20250705101958_00004/tv_segment_9_20250801_145631_00000400-00000449.mp4"
with open(video, 'rb') as f:
  vr = VideoReader(f, ctx=cpu(0))
print('video frames:', len(vr))
# for i in range(len(vr)):
#     # the video reader will handle seeking and skipping in the most efficient manner
#     frame = vr[i]
#     print(frame.shape)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
with torch.inference_mode(), get_safe_autocast():
    inference_state = predictor.init_state(video)
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # add new prompts and instantly get the output on the same frame
    # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
    # sending all clicks (and their labels) to `add_new_points_or_box`
    points = np.array([[700, 600], [690, 620]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)

    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=ann_frame_idx,
                                                                              obj_id=ann_obj_id, points=points,
                                                                              labels=labels)

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(vr[ann_frame_idx])
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.show()
    # propagate the prompts to get masklets throughout the video
    # for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    #     ...

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 5
    plt.close("all")
    for out_frame_idx in range(0, len(vr), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(vr[out_frame_idx])
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.show()
