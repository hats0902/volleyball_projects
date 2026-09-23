# 2026-09-21: `finetune_results/train_54`のbest.ptが何エポック目の重みか調査

## やったこと

ユーザーが投入した`finetune_results/train_54`（`yolo26n-pose`のファインチューニング結果、200エポック）について、`weights/best.pt`が何エポック目の重みかを特定した。`best.pt`単体のメタデータだけでは判断できなかったため、インストール済みultralyticsライブラリのソースコードを確認した。

## 調査方法・根拠

1. `torch.load('weights/best.pt', weights_only=False)`でチェックポイントを直接読み込み。`epoch: -1`, `best_fitness: None`だった
2. なぜこうなるかを`/opt/conda/lib/python3.12/site-packages/ultralytics`（インストール済みバージョン8.4.138。学習に使われたのは`args.yaml`記載の8.4.157で、微妙にバージョンが異なる点に注意）のソースで確認:
   - `ultralytics/utils/torch_utils.py`の`strip_optimizer()`（801〜858行目付近）が、学習終了時に`best.pt`/`last.pt`に対して自動実行され、`optimizer`・`best_fitness`・`ema`・`updates`・`scaler`を`None`に、`epoch`を`-1`に明示的に上書きしていることを確認（855〜857行目: `for k in "optimizer", "best_fitness", "ema", "updates", "scaler": x[k] = None` / `x["epoch"] = -1`）
   - 一方、`train_metrics`フィールドはこの処理で触られておらず、保存時点の実測値（`metrics/mAP50-95(B)`, `metrics/mAP50-95(P)`, `fitness`等）がそのまま残っていることを確認
3. `train_metrics`から得た`mAP50-95(B)=0.69317`, `mAP50-95(P)=0.75079`, `fitness=1.44396`（＝両者の和。poseタスクのfitness定義と一致。`utils/metrics.py`の`PoseMetrics.fitness`参照）を、`results.csv`の全200エポック分と突き合わせ
4. 一致するエポックは153・154の2つ（表示桁数で同値）。どちらの重みが実際に残ったかを判定するため、`ultralytics/engine/trainer.py`の学習ループの処理順序を確認:
   - 607行目: `self.metrics, self.fitness = self.validate()` — 毎エポック`self.fitness`を再計算
   - `validate()`内部（881行目付近）: `self.best_fitness`は**厳密に上回った場合のみ**更新（`if self.best_fitness is None or self.best_fitness < fitness: self.best_fitness = fitness`）
   - 775行目（`save_model()`内）: `if self.best_fitness == self.fitness: self.best.write_bytes(...)` — **同率でも保存（上書き）される**
   - この順序から、153エポック目で`best_fitness`が更新された後、154エポック目でも同じ値のためこの条件が再び真になり、`best.pt`が154エポック目の重みで**再度上書きされる**と判断

## 結論

`finetune_results/train_54/weights/best.pt`は**154エポック目**（全200エポック中）の重み。153エポック目と154エポック目が指標上は同率1位だったが、後から実行された154エポック目の保存で上書きされて最終的に残った。

## 注意点・未検証事項

- ultralyticsのバージョン差（学習時8.4.157 vs 確認時8.4.138）による`strip_optimizer()`・学習ループの挙動差は未検証。パッチバージョン差なので変更はないと想定しているが、断定はできない
- 153/154エポックの`mAP50-95(B)`/`mAP50-95(P)`は`results.csv`の表示桁数（小数点5桁）で完全一致しているだけで、より高精度な内部値まで完全一致しているかは未確認（コード上の挙動から154エポック目と判断したロジックの結論自体には影響しない）
