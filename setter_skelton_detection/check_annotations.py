"""coco-annotatorで作成したkeypointsアノテーション(setter_keypoints.json)の
妥当性を確認するスクリプト。

チェック項目:
  1. bboxが画像範囲(width/height)からはみ出していないか
  2. 1画像あたりのbbox数が1個かどうか
  3. 1画像あたりのkeypoint数(visibility>0のもの)

使い方:
  python check_annotations.py
  python check_annotations.py --json path/to/setter_keypoints.json --images-dir path/to/images
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_JSON = Path(__file__).parent / "setter_keypoints.json"
DEFAULT_IMAGES_DIR = Path(__file__).parent / "coco-annotator" / "datasets" / "setter_keypoints"


def load_annotations(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def check_bbox_overflow(images_by_id, annotations):
    """bboxが画像範囲(0,0)-(width,height)からはみ出していないか確認する。"""
    issues = []
    for ann in annotations:
        image = images_by_id.get(ann["image_id"])
        if image is None or "bbox" not in ann:
            continue
        x, y, w, h = ann["bbox"]
        width, height = image["width"], image["height"]
        over_left = max(0, -x)
        over_top = max(0, -y)
        over_right = max(0, (x + w) - width)
        over_bottom = max(0, (y + h) - height)
        if over_left or over_top or over_right or over_bottom:
            issues.append(
                {
                    "file_name": image["file_name"],
                    "image_id": image["id"],
                    "annotation_id": ann["id"],
                    "bbox": [x, y, w, h],
                    "image_size": [width, height],
                    "over_left": over_left,
                    "over_top": over_top,
                    "over_right": over_right,
                    "over_bottom": over_bottom,
                }
            )
    return issues


def check_bbox_count_per_image(images, annotations_by_image):
    """1画像あたりのbbox数が1個かどうか確認する。"""
    issues = []
    for image in images:
        anns = annotations_by_image.get(image["id"], [])
        bbox_count = sum(1 for a in anns if "bbox" in a)
        if bbox_count != 1:
            issues.append(
                {
                    "file_name": image["file_name"],
                    "image_id": image["id"],
                    "bbox_count": bbox_count,
                }
            )
    return issues


def count_keypoints(ann, num_keypoint_names):
    """keypoints配列(x,y,v の三つ組)からvisibility>0の個数を数える。"""
    keypoints = ann.get("keypoints", [])
    visible = 0
    for i in range(num_keypoint_names):
        v = keypoints[i * 3 + 2] if i * 3 + 2 < len(keypoints) else 0
        if v and v > 0:
            visible += 1
    return visible


def build_keypoint_report(images, annotations_by_image, num_keypoint_names):
    """1画像あたりのkeypoint数を集計する(bboxが複数/0の画像は全annotation分を列挙)。"""
    rows = []
    for image in images:
        anns = annotations_by_image.get(image["id"], [])
        if not anns:
            rows.append(
                {
                    "file_name": image["file_name"],
                    "image_id": image["id"],
                    "annotation_id": None,
                    "num_keypoints_recorded": None,
                    "num_keypoints_visible": 0,
                }
            )
            continue
        for ann in anns:
            visible = count_keypoints(ann, num_keypoint_names)
            rows.append(
                {
                    "file_name": image["file_name"],
                    "image_id": image["id"],
                    "annotation_id": ann["id"],
                    "num_keypoints_recorded": ann.get("num_keypoints"),
                    "num_keypoints_visible": visible,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="setter_keypoints.jsonのパス")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="アノテーション対象の画像が置かれているディレクトリ(存在チェックに使用)",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path(__file__).parent / "annotation_check_report.csv",
        help="keypoint数などをまとめたCSVレポートの出力先",
    )
    args = parser.parse_args()

    data = load_annotations(args.json)
    images = data["images"]
    annotations = data["annotations"]
    keypoint_names = data["categories"][0]["keypoints"]
    num_keypoint_names = len(keypoint_names)

    images_by_id = {im["id"]: im for im in images}
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[ann["image_id"]].append(ann)

    print(f"対象画像数: {len(images)}, アノテーション数: {len(annotations)}, keypoint種類数: {num_keypoint_names}")
    print()

    # 1. bboxが画像範囲外にはみ出していないか
    bbox_overflow_issues = check_bbox_overflow(images_by_id, annotations)
    print(f"[1] bboxが画像範囲外にはみ出しているケース: {len(bbox_overflow_issues)}件")
    for issue in bbox_overflow_issues:
        overflow_desc = ", ".join(
            f"{k}={v}"
            for k, v in (
                ("left", issue["over_left"]),
                ("top", issue["over_top"]),
                ("right", issue["over_right"]),
                ("bottom", issue["over_bottom"]),
            )
            if v
        )
        print(
            f"  - {issue['file_name']} (image_id={issue['image_id']}, annotation_id={issue['annotation_id']}): "
            f"bbox={issue['bbox']}, image_size={issue['image_size']}, overflow=({overflow_desc})"
        )
    print()

    # 2. 1画像あたりのbbox数が1個かどうか
    bbox_count_issues = check_bbox_count_per_image(images, annotations_by_image)
    print(f"[2] bbox数が1個でない画像: {len(bbox_count_issues)}件")
    for issue in bbox_count_issues:
        print(f"  - {issue['file_name']} (image_id={issue['image_id']}): bbox_count={issue['bbox_count']}")
    print()

    # 3. 1画像あたりのkeypoint数
    keypoint_rows = build_keypoint_report(images, annotations_by_image, num_keypoint_names)
    counts = [row["num_keypoints_visible"] for row in keypoint_rows]
    print(f"[3] keypoint数(visibility>0)の分布: min={min(counts)}, max={max(counts)}, 種類数={num_keypoint_names}")
    mismatches = [
        row
        for row in keypoint_rows
        if row["num_keypoints_recorded"] is not None and row["num_keypoints_recorded"] != row["num_keypoints_visible"]
    ]
    if mismatches:
        print(f"  ※ num_keypoints(JSON記載値)と実カウントが不一致: {len(mismatches)}件")
        for row in mismatches:
            print(
                f"    - {row['file_name']} (annotation_id={row['annotation_id']}): "
                f"recorded={row['num_keypoints_recorded']}, actual={row['num_keypoints_visible']}"
            )
    zero_keypoint_rows = [row for row in keypoint_rows if row["num_keypoints_visible"] == 0]
    print(f"  ※ keypointが1つも打たれていない画像/アノテーション: {len(zero_keypoint_rows)}件")
    for row in zero_keypoint_rows:
        print(f"    - {row['file_name']} (image_id={row['image_id']}, annotation_id={row['annotation_id']})")
    print()

    # CSVレポート出力(全件のkeypoint数一覧。目視確認用)
    with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "image_id", "annotation_id", "num_keypoints_recorded", "num_keypoints_visible"],
        )
        writer.writeheader()
        writer.writerows(keypoint_rows)
    print(f"keypoint数の一覧をCSVに出力しました: {args.report_csv}")

    total_issues = len(bbox_overflow_issues) + len(bbox_count_issues) + len(mismatches)
    if total_issues:
        print(f"\n合計 {total_issues} 件の問題が見つかりました。")
    else:
        print("\n問題は見つかりませんでした。")


if __name__ == "__main__":
    main()
