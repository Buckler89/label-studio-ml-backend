import os
# if using Apple MPS, fall back to CPU for unsupported ops
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import tempfile
import cv2


def show_mask(mask: np.ndarray, ax: plt.Axes | np.ndarray, obj_id: int | None = None, random_color: bool = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if isinstance(ax, np.ndarray):
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
        ax = cv2.addWeighted(ax, 0.5, mask_image, 0.5, 0)
    elif isinstance(ax, plt.Axes):
        ax.imshow(mask_image)
    return ax


def show_points(coords: np.ndarray, labels, ax: plt.Axes | np.ndarray, marker_size: int = 200) -> plt.Axes | np.ndarray:
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    if isinstance(ax, np.ndarray):
        # draw point on the image with opencv
        for p in pos_points:
            cv2.circle(ax, tuple(p), 5, (0, 255, 0), -1)
        for p in neg_points:
            cv2.circle(ax, tuple(p), 5, (0, 0, 255), -1)

    elif isinstance(ax, plt.Axes):
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    return ax

def show_box(box, ax: plt.Axes | np.ndarray) -> plt.Axes | np.ndarray:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    if isinstance(ax, np.ndarray):
        cv2.rectangle(ax, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
    elif isinstance(ax, plt.Axes):
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    return ax

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = "/media/diego/Dati/Projects/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("predictor loaded")
    video_path = "/home/diego/Downloads/20241022-134918-121432_day.mp4"
    # video_path = "/home/diego/Downloads/20241025-145032-937345_radar.mp4"
    prompts = {}  # hold all the clicks we add for visualization

    with tempfile.TemporaryDirectory() as temp_dir:
        # check if video is file or folder
        if os.path.isdir(video_path):
            # assume the video already has been split into frames
            video_dir = video_path
        else:
            # ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
            os.system(f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {temp_dir}/'%05d.jpg'")
            video_dir = temp_dir

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(temp_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # take a look the first video frame
        frame_idx = 0
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
        plt.show()

        inference_state = predictor.init_state(video_path=video_dir)
        print("inference_state initialized")
        # predictor.reset_state(inference_state)
        points1 = np.array([[380, 355]], dtype=np.float32)

        points2 = np.array([[300, 340]], dtype=np.float32)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a positive click at (x1, y1) and a negative click at (x2, y2)
        points = np.concatenate([points1, points2], axis=0)
        # box = np.array([points[0][0]-100, points[0][1]-50, points[0][0]+30, points[0][1]+30], dtype=np.float32)
        # points = np.array([[50, 50], [110, 50], [50, 110], [110, 110]], dtype=np.float32)
        # points = np.array([[80, 80]], dtype=np.float32)
        box = np.array([points[0][0]-5, points[0][1]-10, points[0][0]+5, points[0][1]+10], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1, 0], dtype=np.int64)
        prompts[ann_obj_id] = points, labels

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            # box=box,
        )

        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        # show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        points = np.concatenate([points1, points2], axis=0)
        labels = np.array([0, 1], dtype=np.int64)
        ann_obj_id = 2
        prompts[ann_obj_id] = points, labels

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        # show the results on the current (interacted) frame on all objects
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        for i, out_obj_id in enumerate(out_obj_ids):
            show_points(*prompts[out_obj_id], plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.show()

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # # render the segmentation results every few frames
        # vis_frame_stride = 5
        # plt.close("all")
        # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        #     img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        #     plt.figure(figsize=(6, 4))
        #     plt.title(f"frame {out_frame_idx}")
        #     plt.imshow(img)
        #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        #     plt.show()

        # render the segmentation results every few frames
        vis_frame_stride = 1
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            img = cv2.imread(os.path.join(video_dir, frame_names[out_frame_idx]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                img = show_mask(out_mask, img, obj_id=out_obj_id)
            cv2.imshow("frame", img)
            if cv2.waitKey(1000) == ord('q'):
                break

