# Zenn Content Repository

[![Zenn](https://img.shields.io/badge/Zenn-111111?style=flat&logo=zenn&logoColor=3EA8FF)](https://zenn.dev/natomi1203)

このリポジトリは、[Zenn](https://zenn.dev/) に投稿する記事（Article）や本（Book）のコンテンツを管理するためのものです。GitHub連携機能を利用し、`main` ブランチへのPushをトリガーにZennへの自動デプロイを行っています。

## 📝 概要

主に機械学習、データエンジニアリング、Python（Pandas, Polarsなど）、GCPに関連する技術記事を執筆・管理しています。

## 📁 ディレクトリ構成

- `articles/`: Zennの個別の記事データ（マークダウンファイル）を格納しています。各ファイルは自動的にZennの記事として設定されます。
- `books/`: 本（チャプターごと）の管理。

## 📚 執筆記事

- [データエンジニアリングにおけるパラダイムシフト：Pandas と Polars の構造的比較と性能検証](./articles/pandas-vs-polars.md)
- [【続編】Pandas vs Polars の先へ。1億行のデータを爆速にする GPU × cuDF 活用戦略](./articles/pandas-vs-polars-gpu-cudf.md)
- [LLMによる評価の課題と対策 - LLM as Judge](./articles/llm-as-judge.md)

## 🚀 ワークフロー

1. ローカルまたはブラウザ上で記事の執筆や修正を行います。
2. 変更をCommit / Pushし、Pull Requestを作成します。
3. レビュー後、変更内容を `main` ブランチにマージします。
4. GitHubとZennのリポジトリ連携により、自動的に記事がZenn上に反映（デプロイ）されます。

### 便利な Zenn CLI コマンド

- 新規記事作成: `npx zenn new:article`
- ローカルプレビュー: `npx zenn preview`

## 👤 著者

- **Zenn**: [natomi1203](https://zenn.dev/natomi1203)
- **GitHub**: [@natomi1203](https://github.com/natomi1203)
