# CLAUDE.md（プロジェクトメモリ）

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory (`setter_skelton_detection/`, part of the `volleyball_projects` GitHub repo one level up).

## 概要

バレーボールのセッターの姿勢からトスの位置を予測するモデルを最終目標とする研究プロジェクト。現在の中心作業は、セッターの骨格（keypoints）を高精度に検出するYOLO-poseのファインチューニングと、その成果をHugging Face Spacesのデモとして公開すること。

本プロジェクトは著者本人の自己学習・ポートフォリオ（履歴書記載）としての側面を持つ。次の作業を検討する際は、結果が不確実な小規模研究実験より、既に公開しているデモを実際に良くする・見せられる、という実用性を優先する（`docs/product-requirements.md`参照）。

## プロジェクト構造

### ドキュメントの分類

#### 1. 永続的ドキュメント（`docs/`）

アプリケーション全体の「何を作るか」「どう作るか」を定義する恒久ドキュメント。基本方針が変わらない限り更新しない。

- **[`docs/product-requirements.md`](docs/product-requirements.md)** — プロダクトビジョン、ユーザー、成功の定義、要求事項
- **[`docs/functional-design.md`](docs/functional-design.md)** — システム構成図、データモデル（COCO/YOLO形式）、コンポーネント設計、API
- **[`docs/architecture.md`](docs/architecture.md)** — 技術スタック、ZeroGPU等の技術的制約、パフォーマンス要件
- **[`docs/repository-structure.md`](docs/repository-structure.md)** — 2リポジトリ（本リポジトリ／デモリポジトリ）のディレクトリ構成
- **[`docs/development-guidelines.md`](docs/development-guidelines.md)** — コーディング・命名・テスト・Git規約
- **[`docs/glossary.md`](docs/glossary.md)** — バレーボール用語・ML用語・英日対応表

#### 2. 作業単位のドキュメント（`.steering/[YYYYMMDD]-[作業タイトル].md`）

「今回何をしたか」を記録する変更履歴。1作業＝1ファイル（既存ファイルへの追記はしない）。過去の経緯・意思決定の理由を追う場合はここを参照する。

最新: [`.steering/20260907-hf-demo-fixes-and-next-plan.md`](.steering/20260907-hf-demo-fixes-and-next-plan.md)

### `plan.md`

現在合意している次のイテレーション（データ拡張によるモデル改善）の具体的な作業手順と、検討して見送った代替案をまとめたもの。`docs/`ほど恒久的ではないが、直近の作業計画として`.steering/`より先に参照する。

## 開発プロセス

### ドキュメントを更新すべきタイミング

- 基本方針・要求・アーキテクチャに影響する変更 → 該当する`docs/`配下のファイルを更新
- まとまった作業を1つ完了した → `.steering/[YYYYMMDD]-[作業タイトル].md`を新規作成して記録（既存ファイルは編集しない）
- 次の作業計画が変わった → `plan.md`を更新

### 実装の進め方（このリポジトリでの実態）

このリポジトリに自動ビルド・lint・テストのパイプラインはない（`docs/development-guidelines.md`「テスト規約」参照）。

1. データセット拡張・学習: `Workflow`セクション（下記）に沿ってColab上で実施
2. デモアプリの改修: `pose_estimation_model_for_volleyball`側で変更し、pushする前に`gradio.launch(share=True)`等でローカル（実データ・実推論込み）に動作確認する
3. 変更が一段落したら`.steering/`に記録する

## Live demo（別リポジトリ）

ファインチューニング済みモデルは、Hugging Face Space
[`hats0902/pose_estimation_model_for_volleyball`](https://huggingface.co/spaces/hats0902/pose_estimation_model_for_volleyball)
としてGradioデモを公開している。このSpaceは**独立したgitリポジトリ**（GitHubではなくHF上）で、ローカルでは`volleyball_projects`と同じ階層のsibling checkoutとして存在する（このファイルから見て`../../pose_estimation_model_for_volleyball`）。構造・デプロイ手順・ZeroGPU固有の制約はそちらの`CLAUDE.md`を参照。

過去に`setter_skelton_detection/demo/`としてこのリポジトリ内に同居させていたが、ファイル・`.gitattributes`（LFS）の状態がずれる問題が起き、別リポジトリに分離した経緯がある。`demo/`は再作成しない。

`weights/best.pt`（このリポジトリではgitignored）は、モデルを再学習した際にデモリポジトリ側の`weights/best.pt`と手動で同期する。デモリポジトリへのpushは、そのHFアカウントに登録されたSSH鍵を持つ環境からのみ可能（サンドボックス環境等、鍵がない場合は`Permission denied (publickey)`で失敗する）。

## Workflow（このディレクトリの2つのnotebookにまたがる）

1. **`making dataset.ipynb`**（Colab、`/content/drive/MyDrive/VNL2025`に対して実行）: ボールがセッターの手を離れる瞬間のフレームを抽出し、コート・ネットの位置が画像間で揃うようアフィン変換する
2. **coco-annotator**で手動アノテーションし、`setter_keypoints.json`（COCO形式）を作成する
3. **`finetune_yolopose.ipynb`**（Colab）: COCO形式をYOLO-pose形式に変換し、`yolo11n-pose`をファインチューニングする

両notebookともColab専用（Driveパス、`!pip install`、`%cd /content/`前提）で、ローカルスクリプトとしてそのまま実行することは想定していない。

データ形式の詳細（`setter_keypoints.json`のスキーマ、YOLOラベル形式、`kpt_shape`/`flip_idx`）は`docs/functional-design.md`「データモデル定義」を参照。

## 既知の限界／次のデータ拡張方針

現行モデルは「ボールが手を離れる瞬間（ジャンプ中）」のフレームのみで学習しており、動画でのテストで(1)非ジャンプ姿勢のセッターが検出されない、(2)セッターと紛らわしいポーズの他選手を誤検出することがある、という2つの課題が判明している。次のイテレーションの詳細（フレーム収集・ハードネガティブの追加・再学習・再デプロイの手順）は`plan.md`を参照。

## Annotation validation

`check_annotations.py`が`setter_keypoints.json`のアノテーションミスを検出する。

```bash
python3 check_annotations.py   # setter_keypoints.json と coco-annotator/datasets/setter_keypoints/ を既定で使用
```

bboxの画像範囲外はみ出し、bbox数が1でない画像、`num_keypoints`と実際の`v > 0`の数の不一致を検出し、`annotation_check_report.csv`（目視スポットチェック用）を出力する。`--json`/`--images-dir`/`--report-csv`で対象を変更可能。

## Licensing

`../LICENSE`（リポジトリ最上位）はAGPL-3.0。`ultralytics`（本プロジェクトおよびデモSpaceの両方で使用）自体がAGPL-3.0であり、ネットワーク経由での提供（デモSpace）がAGPL-3.0のソース公開要件を発生させるため。詳細は`README.md`の「ライセンス」セクション参照。

## 注意事項

- `docs/`（永続的）と`.steering/`（作業単位）を混同しない。基本方針の変更は前者、個々の作業記録は後者
- `.steering/`は既存ファイルへの追記ではなく、作業ごとに新規ファイルを作成する
- モデル・デモの変更は、可能な限り実データ・実推論で動作確認してからpushする（`docs/development-guidelines.md`「テスト規約」参照）
- デモリポジトリのコードをこのリポジトリに複製・同居させない
