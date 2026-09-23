# 2026-09-23: mAP50-95の計算方法の深掘り（bbox↔正解ラベルの対応付け、余分な検出の扱い）

## やったこと

READMEに記載した`mAP50-95 (bbox)`等の指標について、「正解bboxが6個の時、検出が7個あった場合に対応付けはどうなるのか、余分な検出は誤差としてどう効いてくるのか」という質問を受け、Ultralyticsのソースコード（インストール済み環境、`engine/validator.py`・`utils/metrics.py`・`models/yolo/pose/val.py`）を読んで計算方法を確認した。

## 1. bboxの対応付け（マッチング）は行われている

単純な個数比較ではなく、IoU（重なり具合）に基づいた**greedyな1対1マッチング**を行っている（`engine/validator.py`の`match_predictions`）。

```python
matches = np.nonzero(iou >= threshold)             # IoU >= 閾値のペアを全部列挙
matches = matches[iou[...].argsort()[::-1]]          # IoUが高い順に並べ替え
matches = matches[np.unique(matches[:, 1], ...)[1]]  # 検出側の重複を除去（1検出=1マッチまで）
matches = matches[np.unique(matches[:, 0], ...)[1]]  # 正解側の重複を除去（1正解=1マッチまで）
```

IoUが高いペアから貪欲に確定させ、1つの正解ラベルには1つの検出しか対応付けない（逆も同様）。

## 2. 正解より多く検出された場合（例: 正解6個、検出7個）

- マッチングできる検出は最大で正解の数（6個）まで
- 7個目の検出は、既にどの正解も他の検出とマッチ済みなら、IoUがどれだけ高くても対応付けられず**false positive（FP）**として扱われる
- 「7個目だから必ずFP」というわけではなく、実際にどの検出がどの正解に一番近いかで決まる（たまたま他の検出より良いマッチであれば、別の検出の方がFPになることもある）

## 3. AP（mAP50-95を構成する1指標）の計算式

`utils/metrics.py`の`ap_per_class`・`compute_ap`。

1. 全画像の検出結果をconfidenceの高い順に並べる
2. 上から累積でTP・FPを数える（`tpc = tp.cumsum()`, `fpc = (1-tp).cumsum()`）
3. 各時点でのprecision・recallを計算:
   - `recall = TP累積 / 正解ラベル総数`
   - `precision = TP累積 / (TP累積 + FP累積)`
4. PR曲線の下の面積 = AP（`mpre`を単調減少包絡線にしてから101点補間で台形積分、COCO方式）
5. これをIoU閾値10段階（`torch.linspace(0.5, 0.95, 10)`、0.50〜0.95を0.05刻み）それぞれで計算し、平均したものが`mAP50-95`

```
mAP50-95 = mean( AP(IoU=0.50), AP(IoU=0.55), ..., AP(IoU=0.95) )
```

## 4. poseの場合はIoUの代わりにOKS

`mAP50-95 (pose)`は、bboxの重なり（IoU）ではなく**OKS（Object Keypoint Similarity）**というキーポイント版の類似度を使って同じ手順（マッチング→PR曲線→AP→平均）を回している（`utils/metrics.py`の`kpt_iou`関数）。

```
OKS = Σ_i [ exp(-d_i^2 / (2 * s^2 * κ_i^2 * A)) * v_i ] / Σ_i v_i
```

- `d_i`: i番目のキーポイントの予測と正解のユークリッド距離
- `A`: 正解bboxの面積（選手が大きく写っているほど許容誤差が緩くなる）
- `κ_i`: キーポイントごとの標準偏差（`OKS_SIGMA`。鼻・目は厳しく、腰・肩は緩めの重み）
- `v_i`: そのキーポイントが正解データで可視だったか

## 5. 「余分な検出が多いとスコアが下がる」という理解の補足

大枠では正しいが、正確には「個数」そのものではなく「**余分な検出がconfidence順でどこに位置するか**」で影響の大きさが変わる。

- precisionは「そのconfidence以上の検出すべて」に対する累積比率なので、**余分な検出のconfidenceが高い（順位が早い）ほど、それより後ろの全検出のprecisionを押し下げ、影響が大きい**
- 逆に、recallが既に1.0（正解を全部拾い終えた）に達した後の低confidenceな余分な検出なら、PR曲線のごく一部（recall≈1.0付近）にしか影響しない
- **recallそのものには影響しない**（正解ラベル数という分母は変わらないため）。下がるのはあくまでprecision経由
- mAPはテストデータ全体の検出をまとめて1本のPR曲線にしてから計算するため、1画像単位で独立にスコアが決まるわけではない

## 参考

READMEの「精度評価（mAP、正解ラベルあり）」セクションの数値（mAP50-95, precision, recall）は、上記の仕組みで算出されたもの。
