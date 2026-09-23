# リポジトリ構造定義書

本プロジェクトは2つの独立したgitリポジトリにまたがる（前身`setter_skelton_detection`と同じ構成。`CLAUDE.md`「Live demo (separate repo)」参照）。

## `volleyball_projects`（GitHub、モノレポ）

```
volleyball_projects/
├── LICENSE                      # AGPL-3.0
├── .gitignore
├── setter_skelton_detection/    # 前身プロジェクト（凍結・参照専用、内容は変更しない）
│   └── ...（前身の`docs/repository-structure.md`参照）
├── player_skelton_detection/    # 本プロジェクトの本体（アクティブに管理）
│   ├── CLAUDE.md                 # このディレクトリ用のプロジェクトメモリ
│   ├── README.md                 # 経緯・評価結果のまとめ（今後作成、評価完了後）
│   ├── plan.md                   # 現在合意している次のイテレーションの計画
│   ├── docs/                     # 永続的ドキュメント（本ファイルもここ）
│   ├── .steering/                # 作業単位の変更履歴（今後、作業のたびに追加）
│   ├── （今後作成）フレーム抽出notebook       # 評価用フレームの抽出
│   ├── （今後作成）評価notebook              # mAP算出・目視スポットチェック
│   ├── （今後作成）finetune_yolopose.ipynb   # ファインチューニングが必要な場合のみ
│   ├── check_annotations.py      # 前身からコピーし、複数bbox対応に改修（新規配置）
│   ├── player_keypoints.json     # アノテーション本体（COCO形式、今後作成）
│   └── weights/                  # 採用モデル（gitignored）
├── volleyball_project/          # 旧イテレーション（gitignored、丸ごと。参照専用）
└── data_analysis/               # 無関係な別ノートブック（大会分析、pose検出とは無関係）
```

### 前身ディレクトリとの共有リソースについて

試合動画（`VNL2025/videos`）とcoco-annotatorのDockerクローンは、前身`setter_skelton_detection/`に既に存在する。動画は大容量の個人所有データであり、コート・選手は同じ試合を撮影したものなので、**`player_skelton_detection/`側には複製せず、`../setter_skelton_detection/VNL2025/videos`・`../setter_skelton_detection/coco-annotator/`をそのまま参照・流用する**（新規フレーム抽出notebookの入力パス、アノテーション作業時のDocker起動先として）。この方針は前身ディレクトリの内容を変更するものではない。

## `pose_estimation_model_for_volleyball`（Hugging Face、独立リポジトリ、既存Spaceを転用）

```
pose_estimation_model_for_volleyball/
├── CLAUDE.md          # このリポジトリ用のプロジェクトメモリ（ZeroGPU固有の注意点等）
├── README.md          # タイトル・説明文を全選手向けに更新予定
├── app.py             # Gradioアプリ本体（predict_image / predict_video + 座標エクスポート + トラッキング）
├── requirements.txt
├── weights/           # 採用モデル（事前学習済み or ファインチューニング済み、git-lfs管理）
└── .gitattributes     # LFS対象ファイルの定義
```

リポジトリ自体は前身と同一（新規作成しない）。中身（`app.py`のモデルパス・UI文言・座標エクスポート機能）を本プロジェクト向けに更新する。

## ディレクトリの役割

- **`player_skelton_detection/docs/`**: アプリケーション全体の「何を作るか」「どう作るか」を定義する恒久ドキュメント。基本方針が変わらない限り更新しない
- **`player_skelton_detection/.steering/`**: 「今回何をしたか」を日付付きファイルで記録する変更履歴。作業のたびに新規ファイルを追加していく（既存ファイルへの追記ではなく、1作業＝1ファイル）
- **`setter_skelton_detection/`**: 本プロジェクトに先行するイテレーション。凍結・参照専用、変更しない（本プロジェクトのdocsから経緯としてリンクされる）
- **`volleyball_project/`**: `setter_skelton_detection/`にも置き換えられた、さらに古い旧イテレーション。参照専用、変更しない

## ファイル配置ルール

- Colabノートブックは`player_skelton_detection/`直下に置く（前身の慣習を踏襲、サブディレクトリに分けない）
- 大容量データ（動画・フレーム画像・アノテーションツール本体）は複製せず、上記の通り`setter_skelton_detection/`側を参照する
- モデル重み（`weights/`配下）はこのリポジトリでは常にgitignored。実体はHF Spaces側リポジトリ（git-lfs管理）にのみコミットする
- `docs/`配下のファイル名・構成は本ファイルに準拠する（新しい永続ドキュメントを追加する場合はここにも追記する）
- デモ側（`pose_estimation_model_for_volleyball`）のコードはこのリポジトリに複製・同居させない（前身での失敗経験を踏襲）
