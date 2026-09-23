# CLAUDE.md（プロジェクトメモリ）

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory (`player_skelton_detection/`, part of the `volleyball_projects` GitHub repo one level up).

## 概要

バレーボールの試合動画を入力すると、コート上の選手（全ポジション）の骨格（keypoints）を検出できるシステムを開発するプロジェクト。前身`../setter_skelton_detection/`ではセッターのみに検出対象を限定していたが、実用性・分析用途としての価値の低さを課題として、対象を全選手に広げてピボットした（`docs/product-requirements.md`「プロダクトビジョンと目的」参照）。前身ディレクトリは凍結・参照専用として残しており、変更しない。

現在の中心作業は、①事前学習済み`yolo26n-pose`を試合動画にそのまま適用した際の精度評価、②評価結果に基づくファインチューニング要否の判断、③デモアプリ（既存のHugging Face Space）の全選手検出・トラッキング・座標エクスポート対応への更新、の3つ。

本プロジェクトは著者本人の自己学習・ポートフォリオ（履歴書記載）としての側面を持つ。次の作業を検討する際は、結果が不確実な小規模研究実験より、既に公開しているデモを実際に良くする・見せられる、という実用性を優先する（`docs/product-requirements.md`参照）。

## プロジェクト構造

### ドキュメントの分類

#### 1. 永続的ドキュメント（`docs/`）

アプリケーション全体の「何を作るか」「どう作るか」を定義する恒久ドキュメント。基本方針が変わらない限り更新しない。

- **[`docs/product-requirements.md`](docs/product-requirements.md)** — プロダクトビジョン、ユーザー、成功の定義、要求事項
- **[`docs/functional-design.md`](docs/functional-design.md)** — システム構成図、データモデル（COCO/YOLO形式、座標エクスポートスキーマ）、コンポーネント設計
- **[`docs/architecture.md`](docs/architecture.md)** — 技術スタック、ZeroGPU等の技術的制約、トラッカー選定・パフォーマンス要件
- **[`docs/repository-structure.md`](docs/repository-structure.md)** — 2リポジトリ（本リポジトリ／デモリポジトリ）のディレクトリ構成、前身との共有リソースの扱い
- **[`docs/development-guidelines.md`](docs/development-guidelines.md)** — コーディング・命名・テスト・Git規約
- **[`docs/glossary.md`](docs/glossary.md)** — バレーボール用語・ML用語・英日対応表

#### 2. 作業単位のドキュメント（`.steering/[YYYYMMDD]-[作業タイトル].md`）

「今回何をしたか」を記録する変更履歴。1作業＝1ファイル（既存ファイルへの追記はしない）。まだ作業を実施していないため、このディレクトリは未作成。作業完了のたびに新規作成する。

### `plan.md`

現在合意している初期イテレーション（汎用モデル評価→ファインチューニング要否判断→デモ移行）の具体的な作業手順と、検討して見送った代替案をまとめたもの。`docs/`ほど恒久的ではないが、直近の作業計画として`.steering/`より先に参照する。

## 開発プロセス

### ドキュメントを更新すべきタイミング

- 基本方針・要求・アーキテクチャに影響する変更 → 該当する`docs/`配下のファイルを更新
- まとまった作業を1つ完了した → `.steering/[YYYYMMDD]-[作業タイトル].md`を新規作成して記録（既存ファイルは編集しない）
- 次の作業計画が変わった → `plan.md`を更新

### 実装の進め方（このリポジトリでの実態）

このリポジトリに自動ビルド・lint・テストのパイプラインはない（`docs/development-guidelines.md`「テスト規約」参照）。

1. 汎用モデル評価: `plan.md`「作業内容」1〜5に沿ってColab上で実施（フレーム抽出→アノテーション→検証→評価→要否判断）
2. （要否判断が「要」の場合のみ）ファインチューニング: `plan.md`「作業内容」6に沿って実施
3. デモアプリの改修: `pose_estimation_model_for_volleyball`側で変更し、pushする前に`gradio.launch(share=True)`等でローカル（実データ・実推論込み）に動作確認する
4. 変更が一段落したら`.steering/`に記録する

## Live demo（別リポジトリ、既存Spaceを転用）

ファインチューニング済み（または評価の結果採用が決まった事前学習済み）モデルは、Hugging Face Space
[`hats0902/pose_estimation_model_for_volleyball`](https://huggingface.co/spaces/hats0902/pose_estimation_model_for_volleyball)
としてGradioデモを公開する。これは前身`setter_skelton_detection`が使っていた**既存のSpace**を転用したものであり、新規に作成したものではない。このSpaceは**独立したgitリポジトリ**（GitHubではなくHF上）で、ローカルでは`volleyball_projects`と同じ階層のsibling checkoutとして存在する（このファイルから見て`../../pose_estimation_model_for_volleyball`）。構造・デプロイ手順・ZeroGPU固有の制約はそちらの`CLAUDE.md`を参照。

デモリポジトリへのpushは、そのHFアカウントに登録されたSSH鍵を持つ環境からのみ可能（サンドボックス環境等、鍵がない場合は`Permission denied (publickey)`で失敗する）。

## Workflow（今後、複数のnotebookにまたがる想定）

1. **評価用フレーム抽出**（Colab、`/content/drive/MyDrive/VNL2025`に対して実行想定）: 試合動画から評価用フレームを抽出する。前身の`making dataset.ipynb`の抽出処理を流用
2. **coco-annotator**（前身のクローンを流用、複製しない）で全選手分の手動アノテーションを行い、`player_keypoints.json`（COCO形式、1画像に複数インスタンス）を作成する
3. **評価notebook**（新規）: 事前学習済み`yolo26n-pose`で推論し、mAP50-95等の定量指標と目視評価で精度を確認する
4. （要否判断が「要」の場合のみ）**ファインチューニング**: 前身の`finetune_yolopose.ipynb`を`yolo26n-pose`ベース・全選手データセット向けに改修して実行する

いずれもColab専用（Driveパス、`!pip install`、`%cd /content/`前提）で、ローカルスクリプトとしてそのまま実行することは想定していない。

データ形式の詳細（`player_keypoints.json`のスキーマ、座標エクスポートのJSON/CSVスキーマ、`kpt_shape`/`flip_idx`）は`docs/functional-design.md`「データモデル定義」を参照。

## 既知の限界／前提（開始時点）

- **評価用アノテーションのコスト**: 1画像に複数選手が写るため、少ない枚数でも延べアノテーション数は前身のフルデータセットに匹敵しうる。`plan.md`「評価計画」に軽量化の選択肢を記載済み
- **トラッキングの限界**: 同一チームはユニフォームが同じで見た目上区別しにくく、ブロック時の密集・接触も頻発するため、`track_id`はID switchが起こりうる（完全な同一性の保証ではない）。`docs/functional-design.md`「検出結果エクスポート」参照
- **対象範囲**: 検出対象は入力画像・動画に映っている選手全員とし、片側チームのみ／両チームが映る、といったカメラアングルの違いは問わない方針（審判・ベンチ・観客は対象外、除外ロジックの要否は評価結果を見て判断）

## Annotation validation

`check_annotations.py`（前身からコピーし、複数bbox対応に改修予定）が`player_keypoints.json`のアノテーションミスを検出する。前身と異なり「bbox数が1でない画像」はエラー扱いしない（本プロジェクトでは正常な状態のため）。

## Licensing

`../LICENSE`（リポジトリ最上位）はAGPL-3.0。`ultralytics`（本プロジェクトおよびデモSpaceの両方で使用）自体がAGPL-3.0であり、ネットワーク経由での提供（デモSpace）がAGPL-3.0のソース公開要件を発生させるため。

## 注意事項

- `docs/`（永続的）と`.steering/`（作業単位）を混同しない。基本方針の変更は前者、個々の作業記録は後者
- `.steering/`は既存ファイルへの追記ではなく、作業ごとに新規ファイルを作成する
- モデル・デモの変更は、可能な限り実データ・実推論で動作確認してからpushする（`docs/development-guidelines.md`「テスト規約」参照）
- デモリポジトリのコードをこのリポジトリに複製・同居させない
- **`../setter_skelton_detection/`は凍結・参照専用**。試合動画（`VNL2025/videos`）・coco-annotatorのDockerクローンはそちらを流用するが、`docs/`・`CLAUDE.md`・`plan.md`等のドキュメント自体は変更しない
