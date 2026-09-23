# 2026-09-23: `setter_skelton_detection/`をgit管理から除外（GitHub上では非表示に）

## やったこと

`position-label-pilot`ブランチで、`setter_skelton_detection/`配下のファイル（`CLAUDE.md`、`docs/`一式、`check_annotations.py`等、計22ファイル）がGitHub上に表示され続けている問題を調査・対処した。

## 経緯・原因

`.gitignore`には既に`setter_skelton_detection`という行が追加されていたが、GitHub（リモートリポジトリ）からは消えていなかった。原因は、`.gitignore`は「これから追加される未追跡ファイル」を無視する仕組みであり、**既にgit管理下（トラッキング済み）になっているファイルには効果がない**ため。`setter_skelton_detection/`配下のファイルは、この行が`.gitignore`に追加される前から既にコミットされていたので、無視設定を追加しただけでは追跡が外れず、以後のpushでも引き続きリモートに反映され続けていた。

## 対処内容

```bash
git rm -r --cached setter_skelton_detection
git commit -m "stop tracking setter_skelton_detection (now gitignored)"
git push origin position-label-pilot
```

コミット`cfa8abd`。`git rm --cached`はgitの追跡から外すだけでローカルのファイルは削除しないため、`setter_skelton_detection/VNL2025`（試合動画・フレーム画像。今後も作業に使用する予定）を含め、ローカルの内容はすべてそのまま残っている。push済みで、GitHub上では`setter_skelton_detection/`が見えなくなった。

## 背景・方針判断

`player_skelton_detection`の方針（全選手検出へのピボット）を優先することになったため、前身プロジェクトである`setter_skelton_detection/`はリモートリポジトリ上で公開・表示され続ける必要がない、という判断による。ローカルの`VNL2025`ディレクトリ（試合動画・フレーム画像）は引き続き作業に使う想定のため、ファイル自体は残している。

## 既知の影響・today不整合（未対応）

`docs/repository-structure.md`・`CLAUDE.md`には、`setter_skelton_detection/`が「前身プロジェクト（凍結・参照専用）」としてリポジトリのディレクトリツリーに存在する前提の記述が残っている（例: `docs/repository-structure.md`のツリー図、「`../setter_skelton_detection/VNL2025/videos`をそのまま参照・流用する」という記述）。これはこのdevcontainer環境（`setter_skelton_detection`が実ファイルとしてローカルに存在し続ける環境）では引き続き正しいが、**GitHubからリポジトリを新規cloneした場合は`setter_skelton_detection/`自体が存在しなくなる**ため、その前提での記述は今後実態と合わなくなる。ドキュメント更新は今回のスコープ外として対応していない。
