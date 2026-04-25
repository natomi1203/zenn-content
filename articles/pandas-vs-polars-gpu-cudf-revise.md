---
title: "【続編】Pandas vs Polars の先へ。1 億行対応の GPU × cuDF 活用戦略"
type: "tech"
topics: ["python", "pandas", "polars", "gpu", "cudf"]
published: false
---

## はじめに

前編では、Python におけるデータ処理基盤の選択肢として、Pandas と Polars の構造的な違いを整理した。

その結論を一言で要約すると、次のようになる。

- **100 万行以下の対話的な分析**では、成熟したエコシステムを持つ **Pandas**
- **100 万行を超える ETL や集計処理**では、並列実行と Lazy API を備えた **Polars**

しかし、実務ではさらにその先、すなわち **数千万行から 1 億行規模**の領域が存在する。この領域では、CPUベースのPolarsですら「十分に高速」とは言い切れず、JoinやGroupByのたびに処理待ち時間が目立つようになる。

そこで登場するのが、NVIDIA RAPIDS による **GPU DataFrame エンジンである cuDF** である。

本稿では、前編の比較枠組みを踏襲しつつ、

- GPUを導入すべきデータ量とワークロードの判断基準
- cuDF にはどのような使い方があるのか
- 1 億行規模で最も安定して速い構成は何か

を、実行モデルとメモリ管理の観点から整理していく。

:::message alert
**⚠️ 本記事の前提条件**
- 対象読者：Pandas の基本的な操作に慣れ、大規模データ処理でボトルネックに直面しているエンジニア
- 前提知識：GPU・CUDA の基本概念、ETL/ELT パイプラインの設計経験
- 検証環境：Linux または WSL2 環境（Polars GPU Engine は macOS 非対応）
:::

---

## GPU 時代の分岐点：なぜ Polars の次に cuDF が必要になるのか

Polars は CPU ベースの DataFrame ライブラリとして非常に優秀であり、Pandas の次に選ぶべき第一候補であることは変わらない。
しかし、データ量がさらに増えると、CPU のマルチスレッド処理だけでは吸収しきれないワークロードが現れる。

典型例は次のような処理である。

- 数千万行から 1 億行規模のログに対する複数キー GroupBy
- 巨大なファクトテーブルとディメンションテーブルの Join
- 特徴量生成のためのウィンドウ集計
- DWH から抽出した大量データに対するバッチ前処理

この段階でのボトルネックは、「API の書きやすさ」ではなく、**メモリ帯域と並列計算性能の絶対量**に移る。
GPU はまさにこの領域で威力を発揮する。

### なぜ GPU は CPU より速いのか：3 つの要因

GPU が大規模データで CPU を凌駕するのは、主に以下の理由による。

**1. メモリ帯域幅の差**
| 構成 | メモリ帯域幅 | 差 |
|------|-------------|-----|
| CPU (DDR5-4800 デュアルチャネル) | 約 76 GB/s | 基準 |
| GPU (NVIDIA L4 GDDR6) | 300 GB/s | **約 4 倍** |
| GPU (NVIDIA A100 HBM2e) | 1,555 GB/s | **約 20 倍** |

CPU は DDR5 メモリを使用するため帯域幅が限られるが、GPU は GDDR6 や HBM（High Bandwidth Memory）を使用し、桁違いの帯域幅を誇る。
これにより、1 億行規模のデータをメモリ上で操作する際、GPU はデータを高速に読み書きできる。

**2. コア数と並列性の差**
| 構成 | コア数 | 同時スレッド数 |
|------|--------|----------------|
| CPU (AMD EPYC 9654) | 96 コア | 約 100 スレッド |
| GPU (NVIDIA L4) | 7,424 CUDA コア | **数万スレッド** |

CPU のコアは複雑な制御処理に最適化された少数の「賢いコア」であるのに対し、GPU のコアは単純な演算を並列実行するための「多数の単純コア」である。
GroupBy や Join のようなデータ並列性の高い処理では、GPU が圧倒的なスループットを発揮する。

**3. 演算融合による中間データ削減**

cuDF は複数の演算を単一の CUDA カーネルに融合（kernel fusion）できる。これにより、中間データを VRAM に書き戻す回数が減り、メモリ帯域幅の消費を抑制できる。

例：`df['c'] = df['a'] + df['b'] * 2` という処理で、CPU ベースの Pandas は各演算ごとに中間配列を生成するが、cuDF は単一の CUDA カーネルで完結させる。

:::message
**💡 専門用語の補足**
- **GPU アクセラレーション**: 本来は画像処理向けに設計された **Graphics Processing Unit（GPU）** の大規模並列計算能力を、表形式データの集計・結合・変換処理に活用すること。
- **cuDF**: NVIDIA RAPIDS の一部として提供される GPU ネイティブな DataFrame ライブラリ。Pandas ライクな API を持ちながら、内部では GPU 上で処理が実行される。
- **Video RAM（VRAM）**: **GPU** が持つ専用メモリ。GPU 処理の速度を支える一方で、CPU メモリより容量が小さいため、大規模データではメモリ管理が極めて重要になる。
- **CUDA カーネル**: GPU 上で実行される並列処理プログラム。数千〜数万のスレッドが同時に動作する。
:::

さらに、メモリ効率の観点では Pandas の `object` 型も無視できない。典型的な文字列列では、値そのものに加えて Python オブジェクトとしての管理情報が乗るため、1 つの短い文字列でも **約 50 バイト規模のオーバーヘッド**が発生しうる。

これが 1 億行で積み上がると、ディスク上では軽く見えるデータでも、実メモリ上では一気に扱いづらくなる。

出典: [Apache Arrow, "Reducing Python String Memory Use in Apache Arrow 0.12"](https://arrow.apache.org/blog/2019/02/05/python-string-memory-0.12/)

重要なのは、**GPUを適用したからといって常に高速であるわけではない**という点である。データ量が小さい場合、GPU への転送コストや起動オーバーヘッドが上回り、かえって CPU より遅くなることもある。

### GPU が逆に遅くなるケース：3 つの失敗パターン

GPU 導入を検討する際、以下の条件では CPU 処理の方が高速になる。これらは実際にベンチマークで観測される現象である。

**パターン 1: データ量が小さすぎる（〜10 万行未満）**

| 処理 | Pandas | Polars CPU | Polars GPU |
|------|--------|------------|------------|
| 10 万行の GroupBy | 5ms | 3ms | 50ms |
| 100 万行の GroupBy | 50ms | 20ms | 30ms |
| 1 億行の GroupBy | 5s | 2s | 0.3s |

※ 数値は AWS g5.2xlarge（L4 24GB）での著者実測値の目安であり、ハードウェア構成やクエリ内容により変動する。

GPU は以下のオーバーヘッドを持つ：
1. **ホスト→デバイス転送**: CPU メモリから VRAM へのデータ転送（PCIe バス経由）
2. **カーネル起動コスト**: GPU カーネルの起動自体に数 ms のレイテンシ
3. **デバイス→ホスト転送**: 結果を CPU に戻す転送

データ量が小さい場合、これらのオーバーヘッドが計算時間を上回る。

**パターン 2: 文字列操作が多い処理**

GPU は数値演算には強いが、文字列処理は苦手とする。理由は：
- 文字列は可変長であり、GPU の並列メモリアクセスパターンと相性が悪い
- 正規表現、複雑なパース処理は CUDA カーネルで効率的に実装するのが困難

例：`str.extract()` や `str.replace()` を多用する処理では、GPU の優位性は小さくなる。

**パターン 3: 複雑な条件分岐を含む処理**

GPU は SIMD（Single Instruction, Multiple Data）アーキテクチャを採用しており、同じ命令を多数のスレッドで並列実行する。
条件分岐（if-else）が多いと、スレッド間で実行パスが分岐し（warp divergence）、並列性が低下する。

例：
```python
# GPU が苦手な処理
df['result'] = df['value'].apply(
    lambda x: 'A' if x > 100 else ('B' if x > 50 else 'C')
)
```

このような処理は、CPU ベースの Polars や Pandas の方が高速な場合がある。

---

## cuDF の全体像：実は 3 つの使い方がある

cuDF は単一のライブラリに見えるが、実務上は少なくとも 3 つの利用パターンに分けて考えた方が理解しやすい。

### 3 つのインターフェース比較



| 比較項目 | Pandas Accelerator | cuDF ネイティブ | Polars GPU Engine |
|----------|--------------------|-----------------|-------------------|
| 操作スタイル | Pandas 互換 | cuDF 独自 API | Polars API |
| 実行モデル | Eager | Eager | Lazy + 最適化 |
| 学習コスト | 最小 | 中程度 | 低〜中 |
| 中間データの扱い | 発生しやすい | 書き方次第 | 抑えやすい |
| 1 億行規模への適性 | △ | ◯ | ◎（※Open Beta） |
| 向いている用途 | 既存 Pandas の高速化 | GPU 専用実装 | 大規模 ETL の有力候補 |

### 1. Pandas Accelerator

既存の Pandas コードを大きく書き換えずに GPU の恩恵を受けたい場合の選択肢である。
RAPIDS の `cudf.pandas` は、可能な演算を GPU で実行しつつ、必要に応じて CPU 側へフォールバックする統合 CPU/GPU 体験として提供されている。

出典: [NVIDIA 技術ブログ, "RAPIDS cuDF、コード変更ゼロで pandas を約 150 倍高速化"](https://developer.nvidia.com/ja-jp/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/)

**利点:**
- Pandas に近い感覚で導入できる
- 既存コードの移行コストが小さい

**限界:**
- Eager execution であるため、中間 DataFrame が増えやすい
- 1 億行規模では VRAM を圧迫しやすい

### 2. cuDF ネイティブ

GPU 向けに最初から処理を組み立てる方法であり、RAPIDS の世界観に深く寄せた実装である。

**利点:**
- GPU を前提とした設計がしやすい
- RAPIDS エコシステムと密接に連携できる

**限界:**
- Polars ほどクエリ最適化の恩恵を受けにくい
- Eager モデルなので、書き方によってはメモリ効率が大きくぶれる

### 3. Polars GPU Engine

前編で扱った Polars の Lazy API を保ったまま、実行エンジンとして GPU を利用するアプローチである。
`collect(engine="gpu")` による GPU 実行、Polars の query optimizer の活用、非対応クエリ時の CPU fallback が特徴である。

出典: [Polars, "GPU acceleration with Polars and NVIDIA RAPIDS"](https://pola.rs/posts/gpu-engine-release/), [RAPIDS, "Polars GPU Engine"](https://rapids.ai/polars-gpu-engine/)

**利点:**
- Polarsの宣言的な書き方を維持できる
- クエリ最適化後に GPU 実行へ落とし込める
- 大規模 ETL において安定した性能が出やすい

**制約（2026 年 4 月時点）:**
- 現在 **Open Beta** であり、全ての Polars 式が GPU 実行に対応しているわけではない
- 対応プラットフォームは **Linux および Windows（WSL2 経由）のみ**。macOS は非対応
- NVIDIA GPU（Compute Capability 7.0 以上）と CUDA 12 環境が必要

**結論:**
1 億行規模といった超大規模データに対しては、**Polars + GPU Engine** が最も有力な候補の一つとなる。ただし、Beta である点とプラットフォーム制約は考慮に入れる必要がある。

### 1 億行規模のベンチマーク目安

GPU 系の公式ベンチマークを見ると、1 億行級に近いワークロードでは性能差が「少し速い」では済まない水準まで広がるケースがある。

| 比較対象 | ワークロード | 公式な示し方 | 読み取れること |
|----------|--------------|--------------|----------------|
| `pandas` vs `cuDF-pandas` | 5GB 規模の join / 高度な groupby ベンチマーク | `up to 150x`、数分級の処理が `1〜2秒` 級まで短縮 | 既存の pandas ワークフローでも、データが大きくなると GPU 化の効果は非常に大きい |
| Polars CPU vs Polars GPU Engine | PDS-H benchmark, 80GB データ | `up to 13x` | Polars の書き味を保ったまま、join や group by を多く含む重いクエリで GPU が効きやすい |

これらは同一条件で横一列に比較した単一ベンチマークではなく、`cuDF-pandas` と `Polars GPU Engine` のそれぞれの公式ベンチマークを要約した目安である。したがって、倍率そのものを厳密比較するというより、**大規模データでは CPU だけの処理より GPU 系の実行モデルが大きな差を生みうる**ことを示す材料として読むのが適切である。

:::message
**💡 ベンチマークの読み方に関する補足**

- **PDS-H** は、Polars 開発チームが TPC-H を基に DataFrame ライブラリ向けに翻案した非公式ベンチマークであり、TPC-H のルールには準拠していない。したがって、TPC-H の公式結果とは直接比較できない。
- 上記の倍率はいずれもツールの**開発元が公表した数値**であり、独立した第三者による検証結果ではない点に留意が必要である。
- ベンチマーク結果は、ハードウェア構成・データの特性・クエリパターンによって大きく変動しうる。
:::

それでも、Pandas と Polars / cuDF の差が「実装の好み」ではなく、**実行モデルとハードウェアの差**として現れることは読み取れる。

出典：[NVIDIA 技術ブログ，"RAPIDS cuDF、コード変更ゼロで pandas を約 150 倍高速化"](https://developer.nvidia.com/ja-jp/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/), [Polars, "GPU acceleration with Polars and NVIDIA RAPIDS"](https://pola.rs/posts/gpu-engine-release/), [RAPIDS, "Polars GPU Engine"](https://rapids.ai/polars-gpu-engine/), [Polars, "PDS-H Benchmark"](https://pola.rs/posts/benchmarks/)

### データタイプによる性能差：数値 vs 文字列

GPU の性能発揮度は、データタイプに大きく依存する。

| データタイプ | GPU の優位性 | 理由 |
|-------------|-------------|------|
| **数値（int32/64, float32/64）** | ◎ 大きい | 固定長でメモリアクセスが予測可能。SIMD 演算との相性が抜群に良い。 |
| **Boolean** | ◎ 大きい | ビットパック可能。メモリ効率が高く、帯域幅を有効活用できる。 |
| **日時（datetime64）** | ◯ 中程度 | 数値として扱えるが、タイムゾーン処理などが絡むと複雑化。 |
| **文字列（string/object）** | △ 小さい | 可変長のためメモリアクセスが不規則。正規表現処理は GPU で効率化しにくい。 |

**実測例（100 万行の文字列列 vs 数値列）**
- 数値列の GroupBy：GPU が CPU の 約 8 倍高速
- 文字列列の GroupBy：GPU が CPU の 約 2 倍高速

※ 検証環境：AWS g5.2xlarge（L4 24GB, CUDA 12）、CPU ベースは m6i.2xlarge（8 vCPU）。Polars 1.0.0 以降での実測値。

文字列処理では、GPU の優位性が小さくなることに注意が必要だ。

### 文字列処理が多い場合の対処法

GPU が文字列処理を苦手とする場合、以下の戦略で性能を改善できる。

**戦略 1: 事前フィルタリングで行数を削減**
```python
import polars as pl

# CPU で事前フィルタリングしてから GPU に送信
df = (
    pl.scan_parquet("large_data.parquet")
    .filter(pl.col("category").is_in(["A", "B", "C"]))  # CPU で絞り込み
    .collect(engine="gpu")  # GPU で数値集計
    .group_by("category")
    .agg(pl.col("amount").sum())
)
```

**戦略 2: 文字列を数値エンコーディングに変換**
```python
# 高カーディナリティの文字列列は、事前に変換しておく
df = (
    pl.scan_parquet("large_data.parquet")
    .with_columns(
        pl.col("product_id").cast(pl.Categorical).cast(pl.UInt32)  # 数値化
    )
    .collect(engine="gpu")
)
```

**戦略 3: 文字列操作は CPU、集計は GPU で分担**
```python
# 複雑な文字列処理は CPU で事前処理
df = (
    pl.scan_parquet("large_data.parquet")
    .with_columns(
        pl.col("text").str.extract(r"(\d+)").cast(pl.Int32)  # CPU で処理
    )
    .collect()  # ここで一度 CPU 実行
    .lazy()
    .group_by("extracted_value")
    .agg(pl.col("amount").sum())
    .collect(engine="gpu")  # 集計は GPU
)
```

**戦略 4: 正規表現を単純なパターンに置き換え**
```python
# ❌ GPU で遅い：複雑な正規表現
df = df.with_columns(pl.col("text").str.replace(r"^[a-z]+-\d{4}-.*$", ""))

# ✅ GPU でも比較的速い：単純なパターン
df = df.with_columns(pl.col("text").str.strip_chars())
```

---

## 本質的な差は「GPU を使うか」ではなく「いつ、何を GPU に送るか」

1 億行規模で性能を左右する本当の論点は、単純な GPU の有無ではない。重要なのは、**不要なデータまで VRAM に送り込まないこと**である。

### Eager execution の課題

Pandas Accelerator や cuDF ネイティブでは、基本的にコードを書いた順に処理が実行される。
これは小規模データではわかりやすいが、大規模データでは次の問題を引き起こす。

- 読み込んだ直後の巨大データがそのまま GPU メモリに展開される
- フィルタ前の不要な列まで保持されやすい
- Join や GroupBy のたびに巨大な中間データが発生しやすい
- 結果として OOM に至りやすい

とくに Join は、計算量が重いだけでなく、ピークメモリも膨らみやすい。cuDF の Join でも、キー照合のための内部データ構造を作るため、概念的には入力表に加えてその作業領域と出力ぶんのメモリが効いてくる。ざっくりした見積もりのイメージとしては、必要メモリは次のように表せる。

```text
M_peak ≈ S_table1 + S_table2 + S_hash_table + S_output
```

※ 上記の式は、libcudf の `hash_join` が内部でハッシュテーブルを構築する挙動（参考: [libcudf `hash_join` API](https://docs.rapids.ai/api/libcudf/stable/classcudf_1_1hash__join)）をもとに、筆者がピークメモリの構成要素を概念的に整理したものであり、ドキュメントに記載された公式ではない。

つまり、入力 2 表のサイズだけでなく、結合のためのハッシュテーブルと出力結果ぶんまで同時に抱えやすい。

1 億行規模では、この一時的な膨張がそのまま VRAM 圧迫につながりうる。Join は「遅い処理」である以上に、**OOM を引き起こしやすい処理**でもある。

### Lazy execution の優位性

一方、Polars GPU Engine は、まずクエリ全体を論理プランとして保持し、その後に最適化を行う。Polars 側の説明でも、optimizer が短い実行時間と低メモリ使用量を目指して IR を最適化した後に cuDF 側へ受け渡す構成が示されている。

出典: [Polars, "GPU acceleration with Polars and NVIDIA RAPIDS"](https://pola.rs/posts/gpu-engine-release/), [RAPIDS, "Polars GPU Engine"](https://rapids.ai/polars-gpu-engine/)

これにより、以下のような最適化が自動的に働く。

1. **述語プッシュダウン**: 先にフィルタできる条件は先に適用
2. **射影プッシュダウン**: 必要な列だけを残す
3. **演算融合**: 連続した変換をまとめて実行
4. **不要な中間データの削減**: VRAM 消費を抑制

:::message
**💡 専門用語の補足**
- **述語プッシュダウン（Predicate Pushdown）**: `WHERE` 句のような絞り込み条件を、できるだけデータ読み込みの手前に押し下げて適用する最適化。不要な行を最初から読まないため、I/O とメモリ使用量を減らせる。
- **射影プッシュダウン（Projection Pushdown）**: 最終結果に必要な列だけを早い段階で選び、不要な列を読み込まない最適化。列数の多いデータほど効果が大きい。
- **演算融合（Operation Fusion）**: 連続する複数の変換や計算を一つの処理としてまとめる最適化。中間データを何度も作らずに済むため、高速化と省メモリ化につながる。
:::

この差は、1 億行規模では単なる実装美学ではなく、**処理が完走するかどうか**を左右する。

データサイズが小さいうちは差が見えにくいが、1 億行規模になるとこの差がそのまま速度差とメモリ差になる。

---

## 1 億行規模での実務的な選定基準

GPU 導入の判断は、単に「最速かどうか」だけでなく、**VRAM 容量・運用コスト・処理の複雑さ**を含めて考える必要がある。

ここでいう GPU は一般論としての GPU 全体を指すのではなく、**実務上の DataFrame 処理基盤として最も整備が進んでいる NVIDIA 系 GPU** を前提としている。理由は、本記事の中心である **RAPIDS / cuDF が NVIDIA の CUDA エコシステム上で発展してきた**ためであり、Polars から GPU 実行を利用する場合も、現実的にはこの系統のソフトウェアスタックを使うことになる。つまり、今回の論点は「GPU メーカーの比較」ではなく、**NVIDIA + CUDA + RAPIDS を前提にしたとき、どの構成が 1 億行規模で最も安定するか**にある。AMD など他ベンダーの選択肢も存在するが、Python の DataFrame ワークロードにおける実務情報、周辺ツール、導入事例の蓄積という点では、現時点では NVIDIA 系が最も議論しやすい。

### GPU リソースの目安

| VRAM 容量 | 代表的な GPU | 想定ユースケース | 評価 |
|-----------|--------------|------------------|------|
| 16GB 前後 | T4 | 軽量な集計、検証用途 | △ |
| 24GB 前後 | L4 | 1 億行級の集計・Join の現実的な本命 | **◎** |
| 40GB 以上 | A100 / H100 | 複雑 Join、多数同時処理、数億行級 | ◯〜◎ |

たとえば、`1億行 × 50 カラム × float32` のデータだけでも、単純計算で約 20GB になる。実際の処理では、ここに一時バッファや Join / GroupBy 用の作業領域も乗るため、**VRAM に収まるかどうか**がそのまま実用性を左右する。

この判断の前提として、NVIDIA 公式の仕様では T4 は `16 GB GDDR6 / PCIe Gen3 x16`、L4 は `24GB / PCIe Gen4 x16 64GB/s` とされている。

出典: [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/), [NVIDIA L4](https://www.nvidia.com/ja-jp/data-center/l4/)

### 実務上の見方

**16GB クラス:**
- 単純な変換やフィルタには使える
- ただし大きな Join や高カーディナリティ GroupBy では厳しい
- 1 億行級では VRAM に収まりきらず、ホストメモリ側への退避や転送に依存しやすい
- その結果、GPU の計算性能よりもデータ移動が支配的になりやすい

RAPIDS / cuDF 系では、managed memory pool と prefetching により GPU メモリを超えるデータも扱えるが、そのぶん host-device 間のページ移動が増えやすい。なお、RAPIDS 24.12 以降では **Streaming Executor** が実験的に導入されており、VRAM を超えるデータセットをチャンク単位で処理する仕組みが追加されている。将来的には、VRAM 制約によるOOMの問題がさらに緩和される可能性がある。

出典: [NVIDIA, "Unified Virtual Memory Supercharges pandas with RAPIDS cuDF"](https://developer.nvidia.com/blog/unified-virtual-memory-supercharges-pandas-with-rapids-cudf/), [NVIDIA, "Scaling Up to One Billion Rows of Data in pandas using RAPIDS cuDF"](https://developer.nvidia.com/blog/processing-one-billion-rows-of-data-with-rapids-cudf-pandas-accelerator-mode/)

**24GB クラス:**
- コストと性能のバランスが最も良い
- 1 億行規模のバッチ前処理における「現実解」の一つになりやすい
- 20GB 前後のワークロードを VRAM 内に収めやすく、広帯域メモリを活かしやすい
- L4 は Ada Lovelace 世代で、T4 より新しいアーキテクチャと PCIe Gen4 の恩恵も受けやすい

**40GB 以上:**
- 複雑なワークロードで安定しやすい
- ただしコストが高く、常用するには用途選定が重要

### 結論

コストパフォーマンスを重視するなら、**L4 クラスの GPU に Polars の Lazy API を組み合わせる**構成は、性能・メモリ余裕・運用現実性のバランスを取りやすい有力な選択肢である。

### コスト比較：クラウドインスタンス利用時の目安

GPU 導入を検討する際、コスト対効果は無視できない要素だ。主要クラウドプロバイダのオンデマンドインスタンス単価を比較する。

| インスタンスタイプ | GPU | vCPU | メモリ | 時間単価（目安） | 用途 |
|-------------------|-----|------|--------|------------------|------|
| **AWS g5.xlarge** | L4 | 4 | 16GB | $0.52 | 小〜中規模 GPU 処理 |
| **AWS g5.2xlarge** | L4 | 8 | 32GB | $1.01 | 1 億行級バッチ処理 |
| **AWS g4dn.xlarge** | T4 | 4 | 16GB | $0.53 | 検証・軽量用途 |
| **GCP g2-standard-8** | L4 | 8 | 32GB | $0.45 | コスト重視 |
| **Azure Standard_NC4as_T4_v3** | T4 | 4 | 28GB | $0.55 | 検証用途 |

**CPU のみのインスタンスとの比較**
| インスタンスタイプ | vCPU | メモリ | 時間単価（目安） |
|-------------------|------|--------|------------------|
| AWS m6i.2xlarge | 8 | 32GB | $0.38 |
| AWS c6i.4xlarge | 16 | 32GB | $0.49 |

**ROI（投資対効果）の考え方**

GPU インスタンスは CPU のみのインスタンスと比べて**約 2 倍の単価**だが、処理時間が**数分の一**に短縮できれば、トータルコストは低下する。

例：1 億行の ETL を毎日実行する場合
- **CPU（m6i.2xlarge）**: 30 分 × $0.38/時間 = $0.19/日
- **GPU（g5.2xlarge）**: 5 分 × $1.01/時間 = $0.08/日

→ **GPU の方が約 60% コスト削減** かつ、処理完了までの待ち時間も短縮

ただし、データ量が小さく処理時間差が出ない場合や、GPU 非対応の演算が多い場合は、CPU の方が安価になることもある。

---

## 実装モデルの違いをコードで見る

### Eager モデルの発想

```python
import cudf

df = cudf.read_parquet("events.parquet")
df = df[df["event_type"] == "purchase"]
df["amount_with_tax"] = df["amount"] * 1.1
result = df.groupby("user_id").agg({"amount_with_tax": "sum"})
```

この書き方は直感的だが、各段階で中間データが生成されやすい。

### Lazy + GPU Engine の発想

```python
import polars as pl

result = (
    pl.scan_parquet("events.parquet")
    .filter(pl.col("event_type") == "purchase")
    .with_columns((pl.col("amount") * 1.1).alias("amount_with_tax"))
    .group_by("user_id")
    .agg(pl.col("amount_with_tax").sum())
    .collect(engine="gpu")
)
```

この場合、Polars は「何をしたいか」を先に受け取り、最適化可能な形で全体を見渡したうえで GPU 実行に落とし込める。これは Polars が optimizer を通した IR に対して cuDF をフックする設計であることとも整合している。

出典: [Polars, "GPU acceleration with Polars and NVIDIA RAPIDS"](https://pola.rs/posts/gpu-engine-release/)

### 実践テクニック：VRAM不足を回避する「型」の管理

1億行規模のデータを扱う際、VRAM の節約は計算速度以上に重要である。Polars GPU Engine では、不要な列を減らすだけでなく、**各カラムの型を必要十分な幅に抑える**ことでもメモリ消費を下げられる。

たとえば、デフォルトでは `Int64` や `Float64` として扱われるカラムを、精度が許す範囲で `Int32` や `Float32` に落とすだけでも、**対象カラムの VRAM 消費をほぼ半分にできる**。フラグ列も整数のまま保持するのではなく、`Boolean` に寄せた方がよい。

```python
import polars as pl

# 読み込み直後に型を絞り、GPU に送るデータ量を抑える
df = (
    pl.scan_parquet("large_data_100M.parquet")
    .with_columns(
        [
            pl.col("user_id").cast(pl.Int32),      # 64bitから32bitへ
            pl.col("score").cast(pl.Float32),      # 64bitから32bitへ
            pl.col("is_active").cast(pl.Boolean),  # 数値フラグをBooleanへ
        ]
    )
    .collect(engine="gpu")
)
```

とくに 16GB〜24GB クラスの GPU では、このような型管理の有無が「高速に終わるか」だけでなく、**そもそも OOM を起こさず完走できるか**に影響することがある。1億行時代の GPU 活用では、クエリ最適化に加えて、**列の型を意識して VRAM に載せるデータを小さくする**ことが実践上の基本になる。

---

## セットアップ：Polars GPU Engine と cuDF を導入する

Polars GPU Engine を利用するには、`polars[gpu]` をインストールする。これにより、Polars と cuDF を橋渡しする `cudf-polars` パッケージが自動的に導入される。NVIDIA のパッケージインデックスを明示する必要がある。

:::message alert
**⚠️ 前提条件**
Polars GPU Engine は以下の環境でのみ動作する。

| 項目 | 必須条件 |
|------|----------|
| **OS** | Linux または Windows（WSL2 経由）。**macOS は非対応** |
| **GPU** | NVIDIA GPU（Compute Capability 7.0 以上）。例：T4, L4, A100, H100, RTX 3090/4090 |
| **CUDA** | CUDA 12 環境 |
| **ドライバ** | NVIDIA ドライバ 535 以降 |
| **Python** | 3.9 以上 3.12 未満 |
| **RAM** | システムメモリ 16GB 以上（ホスト - デバイス転送用） |

**GPU の確認方法**
```bash
# 接続されている NVIDIA GPU の確認
nvidia-smi

# Compute Capability の確認
python -c "from numba import cuda; print(cuda.gpus[0].compute_capability)"
```
:::

### インストール（Polars GPU Engine を使う場合）

```bash
# pip の場合
pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com

# uv の場合
uv pip install "polars[gpu]" --extra-index-url=https://pypi.nvidia.com
```

### cuDF ネイティブを直接使う場合

cuDF を Polars 経由ではなく直接使いたい場合は、以下のように別途インストールする。

```bash
uv pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
```

### 確認用の最小コード

```python
import polars as pl

df = pl.DataFrame(
    {
        "user_id": [1, 1, 2, 2, 3],
        "amount": [100, 200, 150, 80, 300],
    }
)

result = (
    df.lazy()
    .group_by("user_id")
    .agg(pl.col("amount").sum())
    .collect(engine="gpu")
)

print(result)
```

### GPU 実行の検証方法

Polars GPU Engine では、以下の方法で GPU 実行の有無を確認できる。

```python
import polars as pl
import os

# GPU エンジンによる実行プランの確認
os.environ["POLARS_VERBOSE"] = "1"

result = (
    pl.scan_parquet("events.parquet")
    .filter(pl.col("event_type") == "purchase")
    .group_by("user_id")
    .agg(pl.col("amount").sum())
    .collect(engine="gpu")
)
```

出力例：
```
RUNTIME LOG: plan ran on GPU: true
GPU operations: 3/5 (60%)
Fallback operations: str.extract, window.rank
```

また、`nvidia-smi` コマンドで GPU の使用状況をリアルタイム監視することも有効だ。

```bash
# 1 秒おきに GPU 使用率を更新
watch -n 1 nvidia-smi
```

### 既知の制限とフォールバック挙動

Polars GPU Engine（2026 年 4 月時点）で**CPU フォールバックが発生しやすい演算**：

| 演算タイプ | 具体例 | 備考 |
|-----------|--------|------|
| 複雑な文字列操作 | `str.extract()`, `str.replace()` with regex | 単純な `str.contains()` は GPU 対応 |
| ウィンドウ関数の一部 | `rank()`, `pct_change()` | 基本的な `sum()`, `mean()` は GPU 対応 |
| 条件付き集計 | `pl.when().then().otherwise()` の複雑な入れ子 | 単純な条件は GPU 対応 |
| 外部関数呼び出し | `apply()` で Python 関数を渡す | GPU 非対応。ベクトル化演算で置き換え推奨 |

これらの演算を使用する場合、GPU 実行されない可能性があるため、事前に検証が必要である。

---

## いつ Pandas / Polars / GPU を選ぶべきか

前編の結論を、GPU を含めて更新すると次のようになる。

| シナリオ | 推奨 | 理由 |
|----------|------|------|
| 小〜中規模データの EDA | Pandas | 即時実行が扱いやすく、周辺エコシステムも豊富 |
| 100 万行超の ETL / 集計 | Polars | CPU ベースでも高速で、省メモリ |
| 数千万行〜1 億行の重い Join / GroupBy | Polars + GPU Engine（※Beta） | Lazy 最適化と GPU 実行を両立。Linux / WSL2 のみ |
| 既存 Pandas 資産をまず高速化したい | Pandas Accelerator | 書き換えコストが小さい |
| GPU 前提で RAPIDS を深く使い込む | cuDF ネイティブ | 専用実装に向く |

### 判断の目安

**Pandas のままでよいケース:**
- 対話的な分析が中心
- データ量が比較的小さい（〜100 万行）
- scikit-learn や既存の Pandas 資産との連携が最重要
- 複雑な非定型処理が多く、Eager モデルの方がデバッグしやすい

**Polars で十分なケース:**
- CPU だけでも SLA を満たせる（バッチ時間 10 分以内など）
- 大規模だが、GPU 運用コストまではかけたくない
- Parquet 中心で Lazy 最適化の恩恵が大きい
- macOS 環境で動作する必要がある

**GPU を検討すべきケース:**
- バッチ時間がビジネス上の制約になっている（例：朝 9 時までに完了必須）
- Join / GroupBy がボトルネックである
- 1 億行級データの前処理を繰り返し実行する
- 数値演算が中心で、文字列操作が少ない
- Linux または WSL2 環境で動作可能

### 意思決定フローチャート

```
データ量
    │
    ├─ 100 万行未満 ────────────→ Pandas
    │
    ├─ 100 万行〜1,000 万行 ────→ Polars（CPU）
    │
    └─ 1,000 万行超
            │
            ├─ バッチ時間 < 5 分で完了 ──→ Polars（CPU）で継続
            │
            ├─ 文字列操作が多い ────────→ Polars（CPU）+ 型最適化
            │
            └─ 数値演算中心 + 時間制約 ─→ Polars + GPU Engine
```

---

## 結論：1 億行時代の有力候補は「Polars の書き心地で GPU を使う」こと

前編では、Pandas から Polars への移行は単なる高速化ではなく、**実行モデルの進化**であると述べた。後編で見てきた GPU 活用は、その延長線上にある。

重要なのは、GPU を導入しても Eager モデルのままでは、1 億行規模における中間データ爆発の問題を根本的には解決しにくいことである。真価を発揮するのは、**Lazy execution によって処理全体を最適化したうえで GPU に流し込む構成**である。

Polars GPU Engine は 2026 年 4 月時点で**Open Beta**であり、対応プラットフォームも Linux / WSL2 に限定される。

全ての式が GPU 実行されるわけではないため、本番導入時にはフォールバック挙動の検証が不可欠である。

これらの制約を踏まえた上で、**Polars で習得した宣言的なデータ処理の書き方**のメリットが色濃く出ている。なぜなら、ライブラリ側がGPUアクセラレーションの仕組みを意識しやすくなり、結果的に「高速処理のアプローチ」そのものが、将来のGPU環境でも通用する、という設計思想が最も実務上の有力な指針となるからだ。
