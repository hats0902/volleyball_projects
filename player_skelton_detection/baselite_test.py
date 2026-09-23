"""事前学習済み yolo26n-pose の簡易動作確認スクリプト（アノテーション不要）。

player_skelton_detection/plan.md の「評価計画」で定めた本番評価（player_keypoints.json
を正解ラベルとしたmAP算出）はまだ実施していない。このスクリプトはその前段として、
「ファインチューニングなしの事前学習済みモデルを実際の試合動画にかけると、
どの程度の頻度で・どの程度の信頼度で選手を検出できるか」を素早く定量的に把握するための
簡易チェック（正解ラベルなしで出せる統計のみ）。

注意: --stride が1より大きい場合、model.track() は使わずプレーンな model() を使う。
理由（実際に踏んだ落とし穴）: model.track() のフレーム間対応付け（カルマンフィルタに
よる動き予測）は連続フレームを前提にしており、大きく間引いた状態（例: --stride 60）
で使うと、同一フレームを単体で処理すれば5〜7件検出できるはずの選手が、トラッカーの
内部状態と噛み合わずに1件しか検出されない、という大幅な過小評価が実際に発生した
（.steering/ の該当記録参照）。--stride 1 で連続フレームを処理する場合のみ
model.track() を使い、track_id をフレーム間で意味のある形で付与する。

使い方:
    python baselite_test.py --video path/to/video.MOV --stride 60 --max-frames 200
    python baselite_test.py --video path/to/video.MOV --stride 1 --max-frames 300  # トラッキング確認用
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="入力動画のパス")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="モデル重み（既定: 事前学習済みyolo26n-pose）")
    parser.add_argument(
        "--stride", type=int, default=60, help="Nフレームに1回処理する（既定60=60fps動画で約1秒おき）"
    )
    parser.add_argument("--max-frames", type=int, default=200, help="処理するサンプリング後フレーム数の上限")
    parser.add_argument("--conf", type=float, default=0.25, help="検出の信頼度しきい値")
    parser.add_argument("--out-dir", default="baselite_test_output", help="注釈画像・統計の出力先ディレクトリ")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"動画を開けませんでした: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # --stride 1（連続フレーム）のときだけ model.track() を使う。間引いた状態で
    # トラッキングを使うと検出数が大幅に過小評価される問題を実際に踏んだため
    # （詳細はモジュールdocstring・.steering/参照）。
    use_tracking = args.stride == 1

    per_frame_records = []  # player_skelton_detection の座標エクスポートスキーマ（動画版）のプロトタイプ
    detections_per_frame = []
    all_confidences = []
    inference_times = []
    track_ids_seen = set()

    # 保存する注釈画像は多くても10枚程度に絞る（目視確認用のサンプルなので全部は不要）
    save_every = max(1, args.max_frames // 10)

    frame_idx = 0
    sampled_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride == 0:
            t0 = time.time()
            if use_tracking:
                results = model.track(frame, persist=True, conf=args.conf, verbose=False)
            else:
                results = model(frame, conf=args.conf, verbose=False)
            inference_times.append(time.time() - t0)

            r = results[0]
            n_det = 0 if r.boxes is None else len(r.boxes)
            detections_per_frame.append(n_det)

            detections = []
            if r.boxes is not None and n_det > 0:
                confs = r.boxes.conf.tolist()
                all_confidences.extend(confs)
                ids = (
                    r.boxes.id.tolist() if (use_tracking and r.boxes.id is not None) else [None] * n_det
                )
                kpts_xy = r.keypoints.xy.tolist() if r.keypoints is not None else [[]] * n_det
                kpts_conf = (
                    r.keypoints.conf.tolist() if (r.keypoints is not None and r.keypoints.conf is not None) else None
                )
                for i in range(n_det):
                    track_id = int(ids[i]) if ids[i] is not None else None
                    if track_id is not None:
                        track_ids_seen.add(track_id)
                    keypoints = [
                        {"x": x, "y": y, "confidence": (kpts_conf[i][k] if kpts_conf is not None else None)}
                        for k, (x, y) in enumerate(kpts_xy[i])
                    ]
                    detections.append(
                        {
                            "track_id": track_id,
                            "confidence": confs[i],
                            "bbox": r.boxes.xywh[i].tolist(),
                            "keypoints": keypoints,
                        }
                    )

            per_frame_records.append({"frame_index": frame_idx, "detections": detections})

            if sampled_idx % save_every == 0:
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), r.plot())

            sampled_idx += 1
            if sampled_idx >= args.max_frames:
                break
        frame_idx += 1

    cap.release()

    with open(out_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump(per_frame_records, f, ensure_ascii=False, indent=2)

    print(f"動画: {args.video}")
    print(f"総フレーム数: {total_frames}（{total_frames / fps:.1f}秒 @ {fps:.1f}fps）")
    print(f"処理したサンプリング後フレーム数: {sampled_idx}（{args.stride}フレームに1回）")
    print()
    print("=== フレームあたりの検出数 ===")
    print(f"  平均: {statistics.mean(detections_per_frame):.2f}")
    print(f"  中央値: {statistics.median(detections_per_frame):.1f}")
    print(f"  最小/最大: {min(detections_per_frame)} / {max(detections_per_frame)}")
    zero_frames = sum(1 for n in detections_per_frame if n == 0)
    print(f"  検出0件のフレーム: {zero_frames}/{sampled_idx}（{100 * zero_frames / sampled_idx:.1f}%）")
    print()
    if all_confidences:
        print("=== 検出の信頼度（confidence） ===")
        print(f"  平均: {statistics.mean(all_confidences):.3f}")
        print(f"  最小/最大: {min(all_confidences):.3f} / {max(all_confidences):.3f}")
    else:
        print("=== 検出の信頼度（confidence） ===\n  検出なし")
    print()
    print("=== トラッキング（track_id） ===")
    if use_tracking:
        print(f"  観測されたユニークtrack_id数: {len(track_ids_seen)}")
    else:
        print(
            f"  トラッキング未使用（--stride={args.stride} > 1のため、通常のmodel()を使用）。"
            "\n  --stride 1 で連続フレームを処理すればtrack_idが付与される"
            "（間引いた状態でmodel.track()を使うと検出数が大幅に過小評価される問題が過去にあったため、意図的に無効化している）"
        )
    print()
    print("=== 処理速度 ===")
    print(f"  平均推論時間/フレーム: {statistics.mean(inference_times) * 1000:.1f}ms")
    print(f"  実効fps: {1 / statistics.mean(inference_times):.1f}")
    print()
    print(f"注釈画像・detections.json の保存先: {out_dir}")


if __name__ == "__main__":
    main()
