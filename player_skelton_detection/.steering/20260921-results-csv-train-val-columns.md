# 2026-09-21: results.csvの`train/*`・`metrics/*`・`val/*`列がどのデータに基づくか調査

## やったこと

ユーザーから「`results.csv`はvalidationデータの結果か」と質問され、直前の回答が一部未検証の推測を含んでいたため、`ultralytics`（インストール済み8.4.138）のソースコードで裏付けを取り直した。

## 調査結果・根拠

- `engine/validator.py` 227行目: `self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)` — validatorのdataloaderは`args.split`（`train_54/args.yaml`では`split: val`）が指すデータから作られる
- 同ファイル 241〜266行目: この`self.dataloader`を1周する中で、推論・損失計算（`self.loss`）・`update_metrics()`（precision/recall/mAP等の集計）を同じループ内で実施
- 同ファイル 291行目: `results = {**stats, **trainer.label_loss_items(loss, prefix="val")}` — `stats`（`metrics/precision(B)`等）と損失（`val/box_loss`等）は、同じ検証パスの結果として1つの辞書にまとめられて返される
- `engine/trainer.py` 901行目: `def label_loss_items(self, loss_items=None, prefix="train")` — 学習ループ側では`prefix="train"`のまま呼ばれ、`results.csv`の`train/*`列（学習データの損失）を作る

## 結論

`results.csv`の列は以下の通り、データソースが混在している。

- `train/*`（`train/box_loss`等）: **学習データ**（学習中の順伝播・逆伝播の損失をエポック内で平均）
- `metrics/*`（`metrics/precision(B)`, `metrics/mAP50-95(B)`, `metrics/mAP50-95(P)`等）・`val/*`（`val/box_loss`等）: **検証データ**（`args.split`が指すsplit＝`train_54`では"val"）に対して、毎エポック同じ検証パス内でまとめて計算

`best.pt`の判定に使う`fitness`（`mAP50-95(B) + mAP50-95(P)`）は`metrics/*`由来なので、検証データに基づく値。ただし「`results.csv`＝検証データの結果」と言い切るのは不正確（`train/*`列は学習データの結果のため）。

## 反省点

この調査を行う前の回答（1つ前のやり取り）では、上記のうち`args.yaml`の`split: val`と`results.csv`のヘッダー構成のみ実際に確認済みで、「`train/*`と`val/*`/`metrics/*`の接頭辞がコードのどこで付与されるか」「validatorが実際に`args.split`のデータを使っているか」は未検証のまま、一般的なultralyticsの知識で回答していた。ユーザーに指摘されて初めて実際にソースを確認し、裏付けを取った。
