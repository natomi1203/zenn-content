# zenn-content

Zenn（[zenn.dev](https://zenn.dev)）に投稿する技術記事および技術書を管理・執筆するためのリポジトリです。
GitHub連携を利用し、Markdownベースでの快適な執筆環境と自動デプロイを実現しています。

## 📂 ディレクトリ構成

```text
.
├── articles/       # 記事 Markdown ファイル (.md)
├── books/          # 本（チャプターごと）の管理
├── images/         # 記事内で使用する画像アセット
└── README.md       # このファイル
```

## 🚀 執筆ワークフロー

本リポジトリでは `zenn-cli` を使用したワークフローを推奨しています。

### 1. 新規記事の作成

以下のコマンドを実行すると、`articles` ディレクトリにユニークなスラッグ（ID）を持つ Markdown ファイルが生成されます。

```bash
npx zenn new:article
```

### 2. ローカルプレビュー

執筆中の内容は、ローカルサーバーで実際の表示を確認しながらブラウザで編集できます。

```bash
npx zenn preview
```

- **URL:** [http://localhost:8000](http://localhost:8000)
- ファイルを保存すると自動的にプレビューに反映されます。

### 3. デプロイ（公開）

1. Markdown ファイルの Front Matter にある `published: false` を `true` に変更します。
2. 変更を `main` ブランチへプッシュ（または Pull Request をマージ）します。
3. Zenn の GitHub 連携機能により、自動的に記事が更新・公開されます。

## 💡 便利な Tips

### 画像の挿入
画像は `/images` ディレクトリに配置し、Markdown 内で相対パスで指定するか、GitHub のリポジトリパスを指定して参照します。

### スラッグの命名
`zenn-cli` が自動生成する 14 桁の英数字（例: `a1b2c3d4e5f6g7`）をそのまま使用することを推奨します。

---

## 🛠 メンテナー向け情報

### Zenn CLI の最新化
```bash
npm install zenn-cli@latest
```

### 執筆ガイド
- [Zenn 執筆ガイド](https://zenn.dev/guide)
- [Zenn Markdown 記法一覧](https://zenn.dev/zenn/articles/markdown-guide)
