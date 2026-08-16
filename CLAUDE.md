# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research project to predict where a volleyball setter will place a set, based on the setter's body pose. The current focus (and the only actively version-controlled part of the repo) is `setter_skelton_detection/`: fine-tuning a YOLO-pose model (`yolo11n-pose`) to reliably detect a setter's skeleton, since the stock YOLO-pose model only found the setter's bbox in ~5/12 frames (setters are airborne, arms overhead, and often partially occluded by other players/the net).

`setter_skelton_detection/README.md` has the full writeup (motivation, dataset creation, training results, and next steps — eventually feeding predicted skeletons into a toss-location model).

## Repo layout and what's actually tracked

Only a subset of the working tree is under git — check `.gitignore` before assuming a directory is part of the repo:

- `setter_skelton_detection/` — the active project. Tracked: `README.md`, the two notebooks, `images/plot.png`, `setter_keypoints.json`, and `VNL2025/{frames,affined,videos}/.gitkeep`.
  - `setter_skelton_detection/coco-annotator/` and `setter_skelton_detection/VNL2025/` (actual video/frame/affine-image content) are gitignored — they're large local working data, not repo content.
  - `coco-annotator` is the (gitignored) clone of the [jsbroks/coco-annotator](https://github.com/jsbroks/coco-annotator) tool used to manually annotate keypoints; annotated images live in `coco-annotator/datasets/setter_keypoints/`.
- `volleyball_project/` — an earlier, broader iteration of the same idea. **Entirely gitignored** (see `.gitignore`); nothing under it is tracked. Its README explicitly says `setter_skelton_detection`'s README superseded it for the fine-tuning writeup. Treat it as legacy/reference only unless the user says otherwise.
- `data_analysis/` — unrelated notebook analyzing Nations League tournament data; not connected to the pose pipeline.

## Workflow (spans both notebooks in `setter_skelton_detection/`)

1. **`making dataset.ipynb`** (run in Google Colab against `/content/drive/MyDrive/VNL2025`): extract the video frame at the instant the ball leaves the setter's hands, then affine-transform it so the court net lands at a consistent position across images (corrects for handheld/no-tripod shooting and differing camera angles per match).
2. Manually annotate keypoints/bbox on the resulting images using **coco-annotator**, producing `setter_keypoints.json` (COCO format: top-level `images` / `categories` / `annotations`; `annotations[i].bbox` is `[x, y, w, h]`; `annotations[i].keypoints` is a flat 17×3 array of `x, y, v` per keypoint, COCO ordering — nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles; `v=0` means intentionally left unannotated because that keypoint wasn't visible, not a labeling error).
3. **`finetune_yolopose.ipynb`** (also Colab; installs `ultralytics`, expects `/content/setter_keypoints.json`): converts the COCO-annotator JSON into per-image YOLO-pose label `.txt` files (`class cx cy w h` + 17×(x,y,v), matching `cfg/datasets/coco8-pose.yaml`'s expected format) under a `setter_skelton_dataset/` directory shaped like `dataset.yaml` (see `volleyball_project/setter_skelton_dataset/dataset.yaml` for the shape: `kpt_shape: [17, 3]`, `flip_idx: [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]`), then fine-tunes `yolo11n-pose`.

Both notebooks are written to run in Google Colab (Drive paths, `!pip install`, `%cd /content/`) — they are not meant to be executed as local scripts as-is.

## Annotation validation

`setter_skelton_detection/check_annotations.py` checks `setter_keypoints.json` for common annotation mistakes before it's fed into fine-tuning:

```bash
cd setter_skelton_detection
python3 check_annotations.py   # defaults to setter_keypoints.json and coco-annotator/datasets/setter_keypoints/
```

It flags: bboxes that extend outside the image's `width`/`height`, images with a bbox count other than 1, and mismatches between an annotation's recorded `num_keypoints` and the actual count of keypoints with `v > 0`. It also writes a per-annotation keypoint-count CSV (`annotation_check_report.csv`) for manual spot-checking. Override paths with `--json` / `--images-dir` / `--report-csv`.

There is no build/lint/test tooling in this repo beyond that script — work here is notebook-driven, and correctness is verified by running cells in Colab and by this validation script.
