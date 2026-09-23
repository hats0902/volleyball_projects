# 2026-09-23: キーポイントのvisibility（v=0, 1, 2）が学習でどう使われるかの調査

## やったこと

COCO形式のキーポイントアノテーションが持つ`v`（visibility）フラグ（0=未アノテーション、1=アノテーションあり・不可視、2=アノテーションあり・可視）について、YOLO-poseの学習時に`v=0, 1, 2`それぞれがどう扱われるかを、インストール済みultralyticsのソース（`utils/loss.py`）を読んで確認した。

## 結論

`v`は学習上**「0かどうか」の2値としてしか使われない**。`v=1`（不可視）と`v=2`（可視）は完全に同じ扱いで、区別されない。

- **`v=0`**: マスクが`False`になり、損失計算から完全に除外される
- **`v=1`・`v=2`**: どちらもマスクが`True`になり、座標誤差loss・キーポイント存在判定loss（・RLE loss、YOLO26系）のすべてに、同じ重みで寄与する

## 根拠（ソースコード）

本プロジェクトが使う`yolo26n-pose`の学習loss（`PoseLoss26.calculate_keypoints_loss`、`utils/loss.py:958`）、およびその親クラス`v8PoseLoss.calculate_keypoints_loss`（`utils/loss.py:776`、YOLOv8系）のいずれも、以下の同一ロジック:

```python
kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # 座標の位置loss（OKSベース）

if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
    kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # キーポイント存在判定loss（二値分類）
```

`gt_kpt[..., 2]`はラベルファイルの`v`の値そのもの（3列目）。`!= 0`という条件式が、v=1とv=2をどちらも`True`にまとめてしまう。

### 各lossでの具体的な使われ方

1. **位置loss（`KeypointLoss.forward`、`utils/loss.py:318`）**: 予測座標と正解座標のユークリッド距離をOKSの式で正規化し、`kpt_mask`（0/1）を掛けて平均する。`kpt_mask`は「その点を誤差計算に含めるか」の重みとして機能するが、v=1とv=2で異なる重みが付くことはない

   ```python
   d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
   e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
   return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
   ```

2. **キーポイント存在判定loss（`kpts_obj_loss`）**: モデルが出力する3チャンネル目（keypointごとのconfidence相当）に対し、`kpt_mask.float()`（0.0 or 1.0）を正解ラベルとしてBCE lossをかける。つまりモデルは「v=1」と「v=2」のどちらであっても等しく「存在する（confidence高く出すべき）」と学習する

3. **RLE loss（YOLO26系、`calculate_rle_loss`、`utils/loss.py:916`）**: `target_weights`はキーポイントの種類（鼻・肩・腰...）ごとに固定された重み（`RLE_WEIGHT`）であり、visibilityの値（v=1 vs v=2）による重み分岐はここにもない。マスク自体は同じ`kpt_mask`（v!=0）を使う

### 参考：評価指標（mAP算出）側も同じ扱い

前回（[`.steering/20260923-map-calculation-deep-dive.md`](20260923-map-calculation-deep-dive.md)）確認したOKS（`utils/metrics.py`の`kpt_iou`）でも:

```python
kpt_mask = kpt1[..., 2] != 0  # (N, 17)
```

と、学習時と全く同じ「v!=0」の2値マスクが使われている。つまり**学習・評価の両方で、v=1とv=2の区別は一切保持されない**。

### 反転拡張（flip augmentation）での扱い

`data/augment.py`の`flip_idx`による左右反転時（例: `instances.keypoints[:, params["flip_idx"], :]`）は、17点の並び順を入れ替えるだけで、各点の`v`値自体への変換処理は行っていない（vはx, yと一緒にそのまま該当キーポイントに付随して移動する）。

## 実務上の含意

- アノテーション時に「見えている（v=2）」か「隠れているが推測できる（v=1）」かを厳密に使い分けても、**現状のultralyticsの学習・評価ロジックでは何の差も生まれない**。両方とも「アノテーションされている点」として同じ重みで扱われる
- 差が生まれるのは`v=0`（未アノテーション）にした場合のみ。「見えていないのであえてアノテーションしない」という前身プロジェクト（`setter_skelton_detection`）の運用方針は、この仕組みと整合している（v=1として無理に推測座標を入れるより、v=0で除外する方が、学習・評価双方で「その点は評価対象外」という意図が正しく反映される）
