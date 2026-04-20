---
title: "【続編】Pandas vs Polars の先へ。1億行のデータを爆速にする GPU × cuDF 活用戦略"
emoji: "🚀"
type: "tech"
topics: ["python", "pandas", "polars", "gpu", "cudf"]
published: false
---

## はじめに

前編では、Python におけるデータ処理基盤の選択肢として、Pandas と Polars の構造的な違いを整理した。

その結論を一言で要約すると、次のようになる。

- **100 万行以下の対話的な分析**では、成熟したエコシステムを持つ **Pandas**
- **100 万行を超える ETL や集計処理**では、並列実行と Lazy API を備えた **Polars**

しかし、実務ではさらにその先、すなわち **数千万行から 1 億行規模**の領域が存在する。この領域では、CPU ベースの Polars ですら「十分に速い」とは言い切れず、Join や GroupBy のたびに待ち時間が目立つようになる。

そこで登場するのが、NVIDIA RAPIDS による **GPU DataFrame エンジンである cuDF** である。

本記事では、前編の比較スタイルを踏襲しつつ、

- GPU を使うべきなのはどの規模からか
- cuDF にはどのような使い方があるのか
- 1 億行規模で最も安定して速い構成は何か

を、実行モデルとメモリ管理の観点から整理していく。

---

## GPU 時代の分岐点：なぜ Polars の次に cuDF が必要になるのか

Polars は CPU ベースの DataFrame ライブラリとして非常に優秀であり、Pandas の次に選ぶべき第一候補であることは変わらない。しかし、データ量がさらに増えると、CPU のマルチスレッド処理だけでは吸収しきれないワークロードが現れる。

典型例は次のような処理である。

- 数千万行から 1 億行規模のログに対する複数キー GroupBy
- 巨大なファクトテーブルとディメンションテーブルの Join
- 特徴量生成のためのウィンドウ集計
- DWH から抽出した大量データに対するバッチ前処理

この段階でのボトルネックは、「API の書きやすさ」ではなく、**メモリ帯域と並列計算性能の絶対量**に移る。GPU はまさにこの領域で威力を発揮する。

:::message
**💡 専門用語の補足**
- **GPU アクセラレーション**: 本来は画像処理向けに設計された **Graphics Processing Unit（GPU）** の大規模並列計算能力を、表形式データの集計・結合・変換処理に活用すること。
- **cuDF**: NVIDIA RAPIDS の一部として提供される GPU ネイティブな DataFrame ライブラリ。Pandas ライクな API を持ちながら、内部では GPU 上で処理が実行される。
- **Video RAM（VRAM）**: **GPU** が持つ専用メモリ。GPU 処理の速度を支える一方で、CPU メモリより容量が小さいため、大規模データではメモリ管理が極めて重要になる。
:::

さらに、メモリ効率の観点では Pandas の `object` 型も無視できない。典型的な文字列列では、値そのものに加えて Python オブジェクトとしての管理情報が乗るため、1 つの短い文字列でも **約 50 バイト規模のオーバーヘッド**が発生しうる。

これが 1 億行で積み上がると、ディスク上では軽く見えるデータでも、実メモリ上では一気に扱いづらくなる。

出典: [Apache Arrow, "Reducing Python String Memory Use in Apache Arrow 0.12"](https://arrow.apache.org/blog/2019/02/05/python-string-memory-0.12/)

重要なのは、**GPU を使えば常に速いわけではない**という点である。データ量が小さい場合、GPU への転送コストや起動オーバーヘッドが上回り、かえって CPU より遅くなることもある。

したがって、前編の「Pandas vs Polars」という比較に、後編ではさらに次の判断軸が加わる。

- **小〜中規模・対話的分析**: Pandas
- **大規模・CPU 最適化**: Polars
- **超大規模・待ち時間削減重視**: GPU / cuDF

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
| 1 億行規模への適性 | △ | ◯ | **◎** |
| 向いている用途 | 既存 Pandas の高速化 | GPU 専用実装 | 大規模 ETL の本命 |

### 1. Pandas Accelerator

既存の Pandas コードを大きく書き換えずに GPU の恩恵を受けたい場合の選択肢である。RAPIDS の `cudf.pandas` は、可能な演算を GPU で実行しつつ、必要に応じて CPU 側へフォールバックする統合 CPU/GPU 体験として提供されている。

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

前編で扱った Polars の Lazy API を保ったまま、実行エンジンとして GPU を利用するアプローチである。`collect(engine="gpu")` による GPU 実行、Polars の query optimizer の活用、非対応クエリ時の CPU fallback が特徴である。

出典: [Polars, "GPU acceleration with Polars and NVIDIA RAPIDS"](https://pola.rs/posts/gpu-engine-release/), [RAPIDS, "Polars GPU Engine"](https://rapids.ai/polars-gpu-engine/)

**利点:**
- Polars の宣言的な書き方を維持できる
- クエリ最適化後に GPU 実行へ落とし込める
- 大規模 ETL において最も安定した性能が出やすい

**結論:**
1 億行規模を対象にするなら、**Polars + GPU Engine** は実務上の有力候補の一つである。

### 1 億行規模のベンチマーク目安

GPU 系の公式ベンチマークを見ると、1 億行級に近いワークロードでは性能差が「少し速い」では済まない水準まで広がるケースがある。

| 比較対象 | ワークロード | 公式な示し方 | 読み取れること |
|----------|--------------|--------------|----------------|
| `pandas` vs `cuDF-pandas` | 5GB 規模の join / 高度な groupby ベンチマーク | `up to 150x`、数分級の処理が `1〜2秒` 級まで短縮 | 既存の pandas ワークフローでも、データが大きくなると GPU 化の効果は非常に大きい |
| Polars CPU vs Polars GPU Engine | PDS-H1 benchmark, 80GB データ | `up to 13x` | Polars の書き味を保ったまま、join や group by を多く含む重いクエリで GPU が効きやすい |

これらは同一条件で横一列に比較した単一ベンチマークではなく、`cuDF-pandas` と `Polars GPU Engine` の公式ベンチマークを要約した目安である。したがって、倍率そのものを厳密比較するというより、**大規模データでは CPU だけの処理より GPU 系の実行モデルが大きな差を生みうる**ことを示す材料として読むのが適切である。

それでも、Pandas と Polars / cuDF の差が「実装の好み」ではなく、**実行モデルとハードウェアの差**として現れることは読み取れる。

出典: [NVIDIA 技術ブログ, "RAPIDS cuDF、コード変更ゼロで pandas を約 150 倍高速化"](https://developer.nvidia.com/ja-jp/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/), [Polars, "GPU acceleration with Polars and NVIDIA RAPIDS"](https://pola.rs/posts/gpu-engine-release/), [RAPIDS, "Polars GPU Engine"](https://rapids.ai/polars-gpu-engine/)

---

## 本質的な差は「GPU を使うか」ではなく「いつ、何を GPU に送るか」

1 億行規模で性能を左右する本当の論点は、単純な GPU の有無ではない。重要なのは、**不要なデータまで VRAM に送り込まないこと**である。

### Eager execution の課題

Pandas Accelerator や cuDF ネイティブでは、基本的にコードを書いた順に処理が実行される。これは小規模データではわかりやすいが、大規模データでは次の問題を引き起こす。

- 読み込んだ直後の巨大データがそのまま GPU メモリに展開される
- フィルタ前の不要な列まで保持されやすい
- Join や GroupBy のたびに巨大な中間データが発生しやすい
- 結果として OOM に至りやすい

とくに Join は、計算量が重いだけでなく、ピークメモリも膨らみやすい。cuDF の Join でも、キー照合のための内部データ構造を作るため、概念的には入力表に加えてその作業領域と出力ぶんのメモリが効いてくる。ざっくりした見積もりのイメージとしては、必要メモリは次のように表せる。

出典: [libcudf `hash_join` API](https://docs.rapids.ai/api/libcudf/stable/classcudf_1_1hash__join)

```text
M_peak ≈ S_table1 + S_table2 + S_hash_table + S_output
```

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

RAPIDS / cuDF 系では、managed memory pool と prefetching により GPU メモリを超えるデータも扱えるが、そのぶん host-device 間のページ移動が増えやすい。

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

## セットアップ：`uv` 環境で cuDF を導入する

cuDF は通常の PyPI パッケージとは取得経路が異なることがあり、`uv` を使う場合は NVIDIA のパッケージインデックスを明示するのが安全である。

### インストール

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

もし GPU 実行に未対応の演算が含まれている場合は、環境や設定によって CPU 側へフォールバックすることがある。そのため、本番導入時は「速いかどうか」だけでなく、**どの演算が GPU 実行されているか**も確認するべきである。

出典: [RAPIDS, "Polars GPU Engine"](https://rapids.ai/polars-gpu-engine/)

---

## いつ Pandas / Polars / GPU を選ぶべきか

前編の結論を、GPU を含めて更新すると次のようになる。

| シナリオ | 推奨 | 理由 |
|----------|------|------|
| 小〜中規模データの EDA | Pandas | 即時実行が扱いやすく、周辺エコシステムも豊富 |
| 100 万行超の ETL / 集計 | Polars | CPU ベースでも高速で、省メモリ |
| 数千万行〜1 億行の重い Join / GroupBy | Polars + GPU Engine | Lazy 最適化と GPU 実行を両立 |
| 既存 Pandas 資産をまず高速化したい | Pandas Accelerator | 書き換えコストが小さい |
| GPU 前提で RAPIDS を深く使い込む | cuDF ネイティブ | 専用実装に向く |

### 判断の目安

**Pandas のままでよいケース:**
- 対話的な分析が中心
- データ量が比較的小さい
- scikit-learn や既存の Pandas 資産との連携が最重要

**Polars で十分なケース:**
- CPU だけでも SLA を満たせる
- 大規模だが、GPU 運用コストまではかけたくない
- Parquet 中心で Lazy 最適化の恩恵が大きい

**GPU を検討すべきケース:**
- バッチ時間がビジネス上の制約になっている
- Join / GroupBy がボトルネックである
- 1 億行級データの前処理を繰り返し実行する

---

## 結論：1 億行時代の本命は「Polars の書き心地で GPU を使う」こと

前編では、Pandas から Polars への移行は単なる高速化ではなく、**実行モデルの進化**であると述べた。後編で見てきた GPU 活用は、その延長線上にある。

重要なのは、GPU を導入しても Eager モデルのままでは、1 億行規模における中間データ爆発の問題を根本的には解決しにくいことである。真価を発揮するのは、**Lazy execution によって処理全体を最適化したうえで GPU に流し込む構成**である。

したがって実務的な結論は明快である。

1. **小〜中規模なら Pandas**
2. **大規模 CPU 処理なら Polars**
3. **1 億行級の待ち時間を削るなら Polars + cuDF**

すなわち、1 億行時代の最適解は「Pandas を捨てて最初から GPU に行く」ことではなく、**Polars で身につけた宣言的なデータ処理の書き方を、そのまま GPU 時代へ拡張すること**にある。
