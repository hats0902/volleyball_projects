# 技術仕様書

> **ドラフト**：トラッカーの具体的な選定（下記参照）は評価フェーズでの実測前提の暫定記述。

## テクノロジースタック

### 汎用モデル評価・データセット作成・ファインチューニング（このリポジトリ、Colab上）

| 領域 | 技術 |
| --- | --- |
| 言語 | Python 3（Google Colabランタイム） |
| 実行環境 | Google Colab（GPUランタイム）、Google Drive（`/content/drive/MyDrive/VNL2025`） |
| MLフレームワーク | [Ultralytics](https://github.com/ultralytics/ultralytics)（`yolo26n-pose`、AGPL-3.0） |
| アノテーションツール | [coco-annotator](https://github.com/jsbroks/coco-annotator)（Docker、ローカル起動） |
| データ形式 | COCO（アノテーション、1画像に複数インスタンス） → YOLO-pose形式（ファインチューニングが必要な場合の学習用ラベル） |
| 検証スクリプト | `check_annotations.py`（複数bbox対応に改修。前身から流用） |
| 評価指標 | mAP50-95等（`ultralytics`の`model.val()`相当の仕組みを流用予定） |

### 推論デモアプリケーション（`pose_estimation_model_for_volleyball`、別リポジトリ、既存Spaceを転用）

| 領域 | 技術 |
| --- | --- |
| 言語 | Python 3 |
| Webフレームワーク | [Gradio](https://www.gradio.app/)（`gr.Blocks`） |
| ホスティング | Hugging Face Spaces（ハードウェア: ZeroGPU） |
| ML/CV | `ultralytics`（推論＋`model.track()`によるトラッキング）、`torch`（ZeroGPU向けCUDA制御）、`opencv-python-headless` |
| 動画処理 | `imageio` + `imageio-ffmpeg`（`libx264`エンコード） |
| ZeroGPU連携 | `spaces`（`@spaces.GPU`デコレータ） |
| 座標データ出力 | 標準ライブラリ（`json`/`csv`）、`gr.File`でダウンロード提供 |

## 開発ツールと手法

- **バージョン管理**: 前身と同様、このリポジトリ（`volleyball_projects`）はGitHub。デモリポジトリ（`pose_estimation_model_for_volleyball`）はHugging Face上の独立したgitリポジトリ（SSH経由でのみpush可能、`git-lfs`で`*.pt`/`*.pth`を管理）
- **開発フロー**: notebook駆動。前身と異なり、まず評価notebookで事前学習済みモデルの精度を確認するフェーズを挟んでからファインチューニングの要否を決める（`docs/functional-design.md`「システム構成図」参照）
- **品質チェック**: 自動lint/型チェックは導入しない（前身踏襲）。アノテーションの妥当性は改修後の`check_annotations.py`で検証。モデル精度はmAP等の定量指標＋目視確認の両方で検証
- **ローカル動作確認**: デモの改修時は、`gradio.launch(share=True)`で一時的な公開URLを発行し、実際の推論込みで動作確認してからHF Spacesにpushする運用を踏襲する

## 技術的制約と要件

前身から引き継ぐ制約:

- **AGPL-3.0ライセンス**: `ultralytics`がAGPL-3.0のため、本プロジェクト・デモアプリともにAGPL-3.0で公開する
- **Hugging Face ZeroGPUの制約**: モデルはモジュールレベルで`.to("cuda")`、`@spaces.GPU`デコレータが最低1つ必要、1日あたりのGPU利用時間はユーザー単位で上限あり
- **git-lfs必須**（デモリポジトリ側）: `*.pt`/`*.pth`を`.gitattributes`から外せない
- **SSH鍵の制約**: デモリポジトリへのpushは、そのHFアカウントに登録されたSSH鍵を持つ環境からしか行えない
- **動画コーデックの制約**: `imageio`＋`libx264`での再エンコードが必須

本プロジェクトで新たに発生する制約:

- **トラッカーの選定**: `ultralytics`は`model.track()`に複数のトラッカー実装（`bytetrack.yaml`、`botsort.yaml`、`ocsort.yaml`、`deepocsort.yaml`、`fasttrack.yaml`、`tracktrack.yaml`）を用意しており、デフォルトは環境のultralyticsバージョンに依存する（現在の開発環境=8.4.138では`tracktrack.yaml`）。BoT-SORT/DeepOCSORT等の見た目照合（ReID）を使うトラッカーは追加のGPU推論コストが発生するが、同一チームの選手は同じユニフォームで見た目上区別しにくいため、コストに見合う精度向上が得られるかは不明。**評価フェーズで実際の試合動画に対して複数トラッカーを試し、精度とGPU処理時間のバランスで選定する**（`docs/functional-design.md`「検出結果エクスポート」の既知の限界も参照）
- **トラッカー状態のリセット**: デモアプリの`model`はモジュールレベルのグローバルオブジェクトであり、`predict_video`はリクエストのたびに同じ`model`インスタンスを再利用する。`model.track(..., persist=True)`はこのオブジェクトにトラッカーの内部状態を保持し続けるため、**何も対策しないと、前のユーザー・前の動画のトラッキング状態が次の動画に引き継がれてしまう**（`track_id`の採番が0から始まらない程度の問題で済む場合もあれば、稀に無関係な動画同士でトラックが誤って結びつく可能性もゼロではない）。各`predict_video`呼び出しの先頭で明示的にトラッカーをリセットする実装が必要（具体的なAPI・挙動はローカル検証で確認してから確定する）
- **多人数検出によるZeroGPU処理時間への影響**: 推論自体（1フレーム1回のforward pass）への影響は人数に依存せず限定的と見込まれるが、`.plot()`の描画コストとトラッキング（対応付け）・座標データへの変換処理はフレームあたりの検出数に比例して増える。前身で設定した`@spaces.GPU(duration=15)`・`ASSUMED_SEC_PER_FRAME`・`MAX_FRAMES`は、複数選手が映る動画で再計測し、必要なら見直す

## パフォーマンス要件

前身の枠組み（`@spaces.GPU(duration=15)`、`MAX_FRAMES`、`max_file_size=300mb`、`height=500`表示）を土台として踏襲しつつ、以下を追加で検証する。

- **複数選手・トラッキングを含めた実測**: `ASSUMED_SEC_PER_FRAME`は前身がセッター1人検出時の暫定値だったため、複数選手検出＋トラッキング処理を含めた値に更新が必要（未実測）
- **トラッカー選定とのトレードオフ**: ReIDベースのトラッカーを採用する場合、1フレームあたりの処理時間増加分を`MAX_FRAMES`の再計算に反映する
