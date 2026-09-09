# リポジトリ構造定義書

本プロジェクトは2つの独立したgitリポジトリにまたがる（`CLAUDE.md`「Live demo (separate repo)」参照）。

## `volleyball_projects`（GitHub、モノレポ）

```
volleyball_projects/
├── LICENSE                      # AGPL-3.0
├── .gitignore
├── setter_skelton_detection/    # 本プロジェクトの本体（唯一アクティブに管理されているディレクトリ）
│   ├── CLAUDE.md                 # このディレクトリ用のプロジェクトメモリ
│   ├── README.md                 # 経緯・データセット作成・学習結果のまとめ（日本語）
│   ├── plan.md                   # 現在合意している次のイテレーションの計画
│   ├── docs/                     # 永続的ドキュメント（本ファイルもここ）
│   ├── .steering/                # 作業単位の変更履歴（日付付きファイル）
│   ├── making dataset.ipynb      # フレーム抽出・アフィン変換（Colab）
│   ├── finetune_yolopose.ipynb   # COCO→YOLO変換・ファインチューニング（Colab）
│   ├── check_annotations.py      # アノテーション検証スクリプト
│   ├── contact_frame_finder.py   # トス瞬間フレーム選定を補助する半自動ヘルパー
│   ├── setter_keypoints.json     # アノテーション本体（COCO形式）
│   ├── images/plot.png           # 学習曲線
│   ├── weights/                  # ファインチューニング済みモデル（gitignored: best.pt）
│   ├── coco-annotator/           # アノテーションツールのclone（gitignored、丸ごと）
│   └── VNL2025/                  # 動画・フレーム・アフィン画像の実データ（gitignored、丸ごと。.gitkeepのみ追跡）
│       ├── videos/.gitkeep
│       ├── frames/.gitkeep
│       └── affined/.gitkeep
├── volleyball_project/          # 旧イテレーション（gitignored、丸ごと。参照専用）
└── data_analysis/               # 無関係な別ノートブック（大会分析、pose検出とは無関係）
```

## `pose_estimation_model_for_volleyball`（Hugging Face、独立リポジトリ）

```
pose_estimation_model_for_volleyball/
├── CLAUDE.md          # このリポジトリ用のプロジェクトメモリ（ZeroGPU固有の注意点等）
├── README.md
├── app.py             # Gradioアプリ本体（predict_image / predict_video）
├── requirements.txt
├── weights/best.pt    # ファインチューニング済みモデル（git-lfs管理）
└── .gitattributes     # LFS対象ファイルの定義
```

## ディレクトリの役割

- **`setter_skelton_detection/docs/`**: アプリケーション全体の「何を作るか」「どう作るか」を定義する恒久ドキュメント。基本方針が変わらない限り更新しない
- **`setter_skelton_detection/.steering/`**: 「今回何をしたか」を日付付きファイルで記録する変更履歴。作業のたびに新規ファイルを追加していく（既存ファイルへの追記ではなく、1作業＝1ファイル）
- **`setter_skelton_detection/coco-annotator/`, `VNL2025/`**: ローカル作業用の実データ。容量が大きく、かつ個人所有の撮影データのため非公開（gitignored）
- **`volleyball_project/`**: 本プロジェクトに置き換えられた旧イテレーション。参照専用、変更しない

## ファイル配置ルール

- Colabノートブックは`setter_skelton_detection/`直下に置く（サブディレクトリに分けない）
- 大容量データ（動画・フレーム画像・アノテーションツール本体）は`.gitignore`に追加し、リポジトリには含めない。ディレクトリの存在だけ`.gitkeep`で示す
- モデル重み（`weights/best.pt`）はこのリポジトリでは常にgitignored。実体はHF Spaces側リポジトリ（git-lfs管理）にのみコミットする
- `docs/`配下のファイル名・構成は本ファイルに準拠する（新しい永続ドキュメントを追加する場合はここにも追記する）
- デモ側（`pose_estimation_model_for_volleyball`）のコードはこのリポジトリに複製・同居させない（過去に`demo/`として同居させて失敗した経緯があるため）
