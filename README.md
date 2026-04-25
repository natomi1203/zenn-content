# Zenn Content Repository

[![Zenn](https://img.shields.io/badge/Zenn-111111?style=flat&logo=zenn&logoColor=3EA8FF)](https://zenn.dev/natomi1203)
[![Repo](https://img.shields.io/badge/Repository-GitHub-181717?style=flat&logo=github)](https://github.com/natomi1203/zenn-content)

このリポジトリは、[Zenn](https://zenn.dev/) に公開する技術記事（Article）や本（Book）のコンテンツを管理するためのものです。  
`main` ブランチにマージされた変更は、GitHub 連携により自動で Zenn に反映されます。

## 概要

主に以下の実践的なテーマを扱っています。

- 機械学習と LLM 評価
- データエンジニアリングと分析ワークフロー
- Pandas / Polars / cuDF などの Python データ処理エコシステム
- クラウド・プラットフォーム設計

## ディレクトリ構成

- `articles/`: Zenn の個別記事（Markdown）
- `books/`: 複数チャプターで構成される Zenn 本

## 執筆記事

- [LLMによる評価の課題と対策 - LLM as Judge](./articles/llm-as-judge.md)
- [【続編】Pandas vs Polars の先へ。1億行のデータを爆速にする GPU × cuDF 活用戦略](./articles/pandas-vs-polars-gpu-cudf.md)
- [データエンジニアリングにおけるパラダイムシフト：Pandas と Polars の構造的比較と性能検証](./articles/pandas-vs-polars.md)

## ワークフロー

1. ローカルで記事を作成・更新する
2. Pull Request を作成してレビューする
3. 承認後に `main` へマージする
4. GitHub 連携により Zenn へ自動反映される

## 便利な Zenn CLI コマンド

- 新規記事作成: `npx zenn new:article`
- ローカルプレビュー: `npx zenn preview`

## 著者

- **Zenn**: [natomi1203](https://zenn.dev/natomi1203)
- **GitHub**: [@natomi1203](https://github.com/natomi1203)
