# 技術仕様書

## テクノロジースタック

### データセット作成・ファインチューニング（このリポジトリ、Colab上）

| 領域 | 技術 |
| --- | --- |
| 言語 | Python 3（Google Colabランタイム） |
| 実行環境 | Google Colab（GPUランタイム）、Google Drive（`/content/drive/MyDrive/VNL2025`） |
| MLフレームワーク | [Ultralytics](https://github.com/ultralytics/ultralytics)（`yolo11n-pose`、AGPL-3.0） |
| アノテーションツール | [coco-annotator](https://github.com/jsbroks/coco-annotator)（Docker、ローカル起動） |
| データ形式 | COCO（アノテーション） → YOLO-pose形式（学習用ラベル） |
| 検証スクリプト | `check_annotations.py`（標準ライブラリのみ、`argparse`/`csv`/`json`） |

### 推論デモアプリケーション（`pose_estimation_model_for_volleyball`、別リポジトリ）

| 領域 | 技術 |
| --- | --- |
| 言語 | Python 3 |
| Webフレームワーク | [Gradio](https://www.gradio.app/)（`gr.Blocks`） |
| ホスティング | Hugging Face Spaces（ハードウェア: ZeroGPU） |
| ML/CV | `ultralytics`（推論）、`torch`（ZeroGPU向けCUDA制御）、`opencv-python-headless`（BGR変換等） |
| 動画処理 | `imageio` + `imageio-ffmpeg`（`libx264`エンコード） |
| ZeroGPU連携 | `spaces`（`@spaces.GPU`デコレータ） |

## 開発ツールと手法

- **バージョン管理**: このリポジトリ（`volleyball_projects`）はGitHub。デモリポジトリ（`pose_estimation_model_for_volleyball`）はHugging Face上の**独立したgitリポジトリ**（SSH経由でのみpush可能、`git-lfs`で`*.pt`/`*.pth`を管理）。両者はコードとしては分離しており、`weights/best.pt`のみ手動で同期する（`CLAUDE.md`参照）
- **開発フロー**: notebook駆動（Colab上でセルを実行して進める）。ビルド・CIパイプラインは存在しない
- **品質チェック**: 自動lint/型チェックは導入していない。アノテーションの妥当性は`check_annotations.py`で検証し、モデル・アプリの動作確認は実際にColab/Spaces上でセルやデモを実行して目視確認する（`CLAUDE.md`「Annotation validation」参照）
- **ローカル動作確認**: デモの改修時は、`gradio.launch(share=True)`で一時的な公開URLを発行し、実際の推論込みで動作確認してからHF Spacesにpushする運用としている（`.steering/`に実施記録あり）

## 技術的制約と要件

- **AGPL-3.0ライセンス**: `ultralytics`がAGPL-3.0のため、本プロジェクト・デモアプリともにAGPL-3.0で公開し、ソースコードを利用可能な状態に保つ必要がある（`../../LICENSE`、デモのFilesタブ）
- **Hugging Face ZeroGPUの制約**:
  - モデルロードはモジュールレベルで`.to("cuda")`する必要がある（`@spaces.GPU`関数内で行うと非効率）
  - `@spaces.GPU`デコレータが最低1つ必要（ないとSpaceが起動しない）
  - 1日あたりのGPU利用時間はユーザー単位（未ログイン2分/日、無料アカウント5分/日、Space横断で共有）で、Space側でコントロールできない
- **git-lfs必須**（デモリポジトリ側）: HFサーバーは非LFSのバイナリpushを拒否するため、`*.pt`/`*.pth`を`.gitattributes`から外せない
- **SSH鍵の制約**: デモリポジトリへのpushは、そのHFアカウントに登録されたSSH鍵を持つ環境からしか行えない
- **動画コーデックの制約**: `cv2.VideoWriter`のmp4v出力はブラウザの`<video>`タグでインライン再生できないため、`imageio`＋`libx264`での再エンコードが必須

## パフォーマンス要件

- **1回あたりのGPU占有時間**: `@spaces.GPU(duration=15)`（15秒）を画像・動画推論の両方に設定。キュー優先度とのトレードオフを考慮した値
- **動画の処理可能フレーム数（`MAX_FRAMES`）**: 未ログインユーザーが1日あたり目安5回の動画テストを枯渇せずに行えるよう、`ASSUMED_SEC_PER_FRAME`（暫定値、実測未検証）から逆算して設定。超過時はエラーまたは間引き処理で対応
- **アップロードサイズ**: `max_file_size=300mb`。誤って大容量動画をアップロードした場合、転送完了を待たずに即エラーとする（ローカル検証で0.1秒程度での即時拒否を確認済み）
- **表示サイズ**: `gr.Image`/`gr.Video`に`height=500`を指定し、縦長画像・動画でも画面に収まるようにする
