# Zenn Content Repository

[![Zenn](https://img.shields.io/badge/Zenn-111111?style=flat&logo=zenn&logoColor=3EA8FF)](https://zenn.dev/natomi1203)
[![Repo](https://img.shields.io/badge/Repository-GitHub-181717?style=flat&logo=github)](https://github.com/natomi1203/zenn-content)

This repository manages content published on [Zenn](https://zenn.dev/), including technical articles and books.
Changes merged into `main` are automatically deployed to Zenn through GitHub integration.

## Overview

The content focuses on practical topics around:

- Machine learning and LLM evaluation
- Data engineering and analytics workflows
- Python ecosystem tools such as Pandas, Polars, and cuDF
- Cloud and platform engineering patterns

## Directory Structure

- `articles/`: Markdown files for individual Zenn articles
- `books/`: Multi-chapter Zenn books
- `images/`: Image assets used in articles and books

## Featured Articles

- [Data Engineering Paradigm Shift: Structural Comparison and Performance Validation of Pandas vs Polars](./articles/pandas-vs-polars.md)
- [Pandas vs Polars Sequel: Accelerating 100 Million Rows with GPU x cuDF](./articles/pandas-vs-polars-gpu-cudf.md)
- [Challenges and Mitigations in LLM-based Evaluation (LLM as Judge)](./articles/llm-as-judge.md)

## Writing Workflow

1. Draft or update content locally.
2. Open a pull request for review.
3. Merge approved changes into `main`.
4. GitHub automatically syncs merged content to Zenn.

## Zenn CLI Commands

- Create a new article: `npx zenn new:article`
- Start local preview: `npx zenn preview`

## Author

- **Zenn**: [natomi1203](https://zenn.dev/natomi1203)
- **GitHub**: [@natomi1203](https://github.com/natomi1203)
