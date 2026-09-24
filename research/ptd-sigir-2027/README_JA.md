# PTD SIGIR 2027 研究進捗

## 現在地（2026-09-25 JST）

リポジトリ規約確認、独立 worktree、論文 scaffold、preregistration、claim--evidence ledger、関連研究比較表、artifact 契約、証跡境界を守った初稿まで完了しています。PTD の one-day tree-family 診断と prospective five-day 評価は **pending** であり、効果の結論には使っていません。

SIGIR 2027 full-paper の公式要項は公開済みです。英語 PDF、最新 ACM `sigconf` 二段組、参考文献を除く最大 9 ページ、匿名査読、CCS concepts と keywords が必要です。確認日時点で日程には "PROPOSED" と表示されています。詳細は `artifact/venue_requirements.md` に固定しています。

## 証跡ポリシー

`VERIFIED` は、固定 URI、manifest、content hash、機械可読 evaluation record がすべて存在する状態です。`PREREGISTERED` は PTD five-day 結果を見る前に固定した方法・分析です。`PENDING` は abstract、結論、結果表、効果主張へ昇格させません。正本は `artifact/claim_evidence_ledger.csv` です。

収録した legacy ESMM anchor は社内制限付き証跡です。teacher 定義と point-in-time-safe な共通評価母集団の固定には使えますが、PTD の有効性を示すものではありません。匿名投稿前に、内容ハッシュを維持した匿名 artifact mirror を作り、組織を特定できる URI は置換します。

## 再現手順

Python 3.10+ と Tectonic（または `acmart` を含む TeX Live）が必要です。

```bash
make all
```

LaTeX の表・図を再生成し、依存なしの PTD 参照実装テストと synthetic smoke check、引用、JSON/CSV、hash、claim status を検証した後、2種類の PDF をビルドします。`paper/main.pdf` は SIGIR 投稿用の匿名査読版、`paper/main-author.pdf` は **NAOYUKI TOMITA**（名前 NAOYUKI、家族名 TOMITA）を紙面に表示する著者版です。個別には `make generate`、`make test`、`make smoke`、`make validate`、`make paper-review`、`make paper-author` を使います。

Makefile は `SOURCE_DATE_EPOCH` を artifact manifest の時刻に固定し、同一入力からの反復ビルドで PDF hash が一致するようにしています。新しい artifact epoch を意図的に発行するときだけ上書きします。

`reference/ptd.py` は sibling distribution、internal-node teacher 集約、temperature-scaled KL objective、deterministic balanced path、capacity-constrained reassignment の実行可能仕様です。production trainer の代替でも、実験結果でもありません。

## PTD 結果を取り込む条件

1. `artifact/evaluation.schema.json` 準拠の immutable evaluation JSON。
2. `artifact/run_manifest.schema.json` 準拠の prospective run manifest と、SHA-256 を含む `artifact/manifest.schema.json` 準拠の研究 artifact manifest。
3. preregistered five-date split、固定 3 seed、test を使った tree 再構築・選択がないこと。
4. point-in-time assertion と candidate-universe check の通過。
5. 8 variants、seed/date 別 metrics、4 registered contrasts、matched latency protocol が揃い、deviation がある場合は自動昇格せず手動監査すること。
6. fail-closed checker が `admissible: true` を返した後、JSON pointer 付き ledger 更新と原稿再生成を行うこと。

読み取り専用の admission check は次で実行します。

```bash
make check-evidence RUN_MANIFEST=/path/to/run_manifest.json EVALUATION=/path/to/evaluation.json
```

checker は immutable hash link、code revision、source/teacher identity、split/seeds、tree-lock timing、metric completeness、bootstrap、Holm 補正 contrast、quality/diversity/latency guardrail を検証します。ledger や原稿は自動編集しません。

## 投稿時の匿名性

この開発リポジトリは所有者を特定できるため、匿名査読原稿から直接リンクできません。証跡 gate 通過後に匿名 snapshot を作ります。所属、謝辞、公開 artifact URL は review/camera-ready の移行まで非表示または placeholder のままです。

原稿ソースの著者名は、名前を NAOYUKI、家族名を TOMITA として **NAOYUKI TOMITA** と記録しています。review build は ACM の `anonymous=true` を維持するため `paper/main.pdf` では実名が非表示ですが、`paper/main-author.pdf` では表示されます。所属は未提示のため推測せず省略しています。
