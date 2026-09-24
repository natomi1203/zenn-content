# PTD SIGIR 2027 研究進捗

## 現在地（2026-09-25 JST）

リポジトリ規約確認、独立 worktree、論文 scaffold、preregistration、claim--evidence ledger、関連研究比較表、artifact 契約、証跡境界を守った初稿まで完了しています。PTD 実行に必要な raw sequence/category 216 shard と frozen teacher checkpoint に加え、結果列を読まない商品監査と機械可読 method contract により、5,584 商品の catalog envelope、日別 mask、実装可能な二系列 encoder、tree/training 設定も固定しました。PTD の one-day tree-family 診断と prospective five-day 評価は **pending** であり、効果の結論には使っていません。

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

`reference/evaluation.py` は binary NDCG/Recall、date-stratified paired bootstrap、percentile interval、two-sided bootstrap sign test、Holm 補正、quality/diversity/latency guardrail を固定します。テストは synthetic 値だけを使い、原稿の結果表には使えません。

`artifact/verified/raw_input_inventory.json` は、ユーザー・商品・系列・ラベル本文を保存せず、GCS generation、checksum、Parquet footer schema、行数を記録します。read-only GCS credential と PyArrow がある環境では `uv run --with pyarrow python scripts/inventory_raw_inputs.py` で再生成でき、SHA-256 は `d28d69f602b3782f923190768b0d2efc64104c456c9b7b961ea457ed04a31db3` のままでなければなりません。

`artifact/verified/raw_sequence_contract.json` は生成 SQL を固定し、click/purchase history が別々の直近順・最大30件の商品 ID 列で、event ごとの timestamp を持たないことを証明します。`artifact/verified/item_universe_audit.json` は件数と hash だけを保存し、train/test の unique item が 3,126/4,592、train・validation にない test item が 2,339、固定 envelope が 5,584 item であることを記録します。後者は `uv run --with pyarrow python scripts/audit_item_universe.py` で再生成でき、SHA-256 は `11293abab08c86bf386963d92c27daf116612d590c1a54d65771bdc442147415` です。

`artifact/preregistered_method.json` は結果を見る前に固定した厳密な method contract（SHA-256 `8fd008bcfddfaeda74f6c6cddfab5b644e664bb5aa645ca778a94a788c5fcfee`）です。time encoding を使わない二系列 HSTU-style model、同じ入力を使う multiwindow-DIN ablation、binary depth-13 tree、eligibility mask、optimizer/sampling budget、L4 runtime を記録します。費用が発生する cloud 実行は、明示的な承認なしには開始しません。

`artifact/verified/execution_readiness_audit.json` は既存 one-day TDM runner の code hash と、これを PTD と呼べない理由を固定します。6つのcore component、すなわち frozen teacher materialization、実5,584商品catalog/date mask、共有二系列HSTU-style/multiwindow-DIN trainer、exact train-only anchored reassignment、five-date/three-seed beam retrieval・latency測定、paired evaluation emitterに決定論的な合成実装証跡があります。`runner/build_training_examples.py` はraw/frozen-teacher rowをkey一致させ、train/validationのcandidate setを固定date maskと照合し、観測purchaseごとに完全なdepth-13 groupを出力します。test shardではoutcome列を開きません。`runner/train_ptd.py` はこのimmutable artifactから登録済みvariant/seedの単一fitを実行し、batch size 64、exact 2 epoch、no-clobber checkpoint、validation-loss auditを強制します。`runner/build_assignment_queries.py`と`runner/materialize_assignment_weights.py`は3つのtrain日だけを隔離し、exact anchored solverが受理する完全なfloat64 train-item×free-leaf行列を生成します。`runner/run_alternating_cycles.py`は一つのalternating variant/seedについてcycle 0--3を完走し、model parameterをwarm-startする一方で両optimizerをresetし、validation purchase NDCG@50とlower-cycle exact-tie ruleで選択して、test queryを開かずに選択bundleを固定します。test側は`runner/build_retrieval_queries.py`と`runner/retrieval_runner.py`が固定24-cell beam/latency matrixまで担当します。これらは実装検査であり、効果、実tree、または本番latencyの証跡ではありません。

有料launchはまだreadyではありません。validation selection、train-only assignment-weight materialization、validation-only alternating-cycle orchestrationは実装済みです。残るcode gateは、完全なfit scheduling・run manifest・retrieval planの組み立てと、完成revisionからの新しい決定論的bundleです。Vertex AI費用の明示承認は、それらとは別の最終gateです。

`artifact/verified/launch_bundle_evidence.json` はrevision `f8158afdc84b8831c1dac6c5c7893e0bec0d5e51` から2回独立生成してbyte一致した `git archive` / `gzip -n` bundle（SHA-256 `ce5c399287192ff5a5a723aa7540b4533e0a90f6906f23634c5ecdea32ce5be9`）を記録します。展開後bundleは利用可能な全テスト、引用検査、artifact検証を通過しました。bundleはlocalにのみ存在し、uploadもVertex AI submitもしていません。

データbridgeと単一fitは意図的に分離され、既存出力があると停止します。

```bash
uv run --with pyarrow --with numpy python runner/build_training_examples.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --output /restricted/training/examples.parquet \
  --manifest /restricted/training/manifest.json

uv run --with torch --with pyarrow --with numpy python runner/train_ptd.py \
  --examples-manifest /restricted/training/manifest.json \
  --variant ptd_combined --seed 16630 \
  --temperature 2 --lambda-item 0.3 --lambda-node 0.3 \
  --checkpoint /restricted/fits/ptd_combined-16630/checkpoint.pt \
  --manifest /restricted/fits/ptd_combined-16630/manifest.json \
  --device cuda

uv run --with pyarrow --with numpy python runner/build_validation_queries.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --output /restricted/validation/queries.jsonl \
  --manifest /restricted/validation/query-manifest.json

uv run --with torch --with pyarrow --with numpy python runner/select_validation.py \
  --validation-queries-manifest /restricted/validation/query-manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --fit-manifest-glob '/restricted/fits/*/manifest.json' \
  --output /restricted/validation/selection.json \
  --device cuda

uv run --with pyarrow --with numpy python runner/build_assignment_queries.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --output /restricted/alternating/train-queries.jsonl \
  --manifest /restricted/alternating/train-query-manifest.json

uv run --with torch --with pyarrow --with numpy --with scipy \
  python runner/materialize_assignment_weights.py \
  --assignment-queries-manifest /restricted/alternating/train-query-manifest.json \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --fit-manifest /restricted/fits/alternating_ptd-16630/manifest.json \
  --output /restricted/alternating/weights.parquet \
  --manifest /restricted/alternating/weight-manifest.json \
  --device cuda

uv run --with numpy --with scipy --with pyarrow python runner/alternating_solver.py \
  --catalog /restricted/catalog/catalog.parquet \
  --date-eligibility /restricted/catalog/date_eligibility.parquet \
  --weights /restricted/alternating/weights.parquet \
  --weight-manifest /restricted/alternating/weight-manifest.json \
  --output-dir /restricted/alternating/tree-cycle-1

uv run --with torch --with pyarrow --with numpy --with scipy \
  python runner/run_alternating_cycles.py \
  --teacher-manifest /restricted/teacher/manifest.json \
  --assignment-queries-manifest /restricted/alternating/train-query-manifest.json \
  --validation-queries-manifest /restricted/validation/query-manifest.json \
  --initial-catalog /restricted/catalog/catalog.parquet \
  --initial-date-eligibility /restricted/catalog/date_eligibility.parquet \
  --variant alternating_ptd --seed 16630 \
  --temperature 2 --lambda-item 0.3 --lambda-node 0.3 \
  --output-dir /restricted/alternating/alternating_ptd-16630 \
  --device cuda
```

個別fitとsolver commandはcomponent単位のentry pointです。cycle orchestratorは一つのalternating variant/seedについて登録済み4-fit chainを実行しますが、prospective実験全体ではありません。6本すべてのalternating chainをscheduleする前に、selectorには27個すべてのcombined-grid manifestと、選択tupleを使ったitem-only/node-only manifestが必要です。

`artifact/vertex_preflight_job_spec.json` はsynthetic runtime互換性検査用で、まだlaunch不能なVertex `CustomJobSpec` templateです。登録済みL4 machine/container、正確なbundle hash、空output/no-clobber検査、container内の `PTD_COST_AUTHORIZED=true` gateを固定し、service accountとGCS locationはplaceholder、authorization既定値はfalseです。`make vertex-preflight-validate` は安全なlocal検証だけを行います。`scripts/validate_vertex_preflight.py` は明示的な `--authorize-cost` と具体的なservice-account/GCS値がある場合だけspecをrenderでき、`gcloud`呼び出しやjob submitは一切行いません。現在、launch可能なrender済みspecは存在しません。

制限付きデータ環境での任意チェックは次です。

```bash
uv run --with torch --with pyarrow --with numpy python scripts/run_teacher_score_smoke.py --checkpoint /path/to/checkpoint-valid_loss.pt
uv run --with torch --with numpy python scripts/run_trainer_smoke.py
uv run --with torch --with numpy python -m unittest tests.test_ptd_model -v
uv run --with numpy --with scipy --with pyarrow python scripts/run_alternating_solver_smoke.py
uv run --with numpy --with scipy --with pyarrow python -m unittest tests.test_alternating_solver -v
uv run --with torch --with pyarrow --with numpy --with scipy --with jsonschema python -m unittest tests.test_alternating_cycles tests.test_train_ptd -v
python3 scripts/run_evaluation_emitter_smoke.py
python3 -m unittest tests.test_evaluation_emitter -v
uv run --with torch --with pyarrow --with numpy python scripts/run_retrieval_runner_smoke.py
uv run --with torch --with pyarrow --with numpy python -m unittest tests.test_build_retrieval_queries tests.test_retrieval_runner -v
uv run --with pyarrow --with numpy python runner/build_catalog_bundle.py --inventory artifact/verified/raw_input_inventory.json --output-dir /restricted/output/catalog-bundle
uv run --with pyarrow --with numpy python runner/build_catalog_bundle.py --inventory artifact/verified/raw_input_inventory.json --output-dir /restricted/output/catalog-bundle-rerun
python3 scripts/record_catalog_bundle_evidence.py --bundle-dir /restricted/output/catalog-bundle --rerun-bundle-dir /restricted/output/catalog-bundle-rerun
```

## PTD 結果を取り込む条件

1. `artifact/evaluation.schema.json` 準拠の immutable evaluation JSON と、`artifact/paired_observation_row.schema.json` に各行が準拠する hash-linked JSONL。
2. `artifact/run_manifest.schema.json` 準拠の prospective run manifest と、正確な raw-input / sequence / item-universe / method-contract / candidate-mask / frozen-teacher hash および `artifact/manifest.schema.json` 準拠の研究 artifact manifest。
3. preregistered five-date split、固定 3 seed、test を使った tree 再構築・選択がないこと。
4. point-in-time assertion と candidate-universe check の通過。
5. 8 variants、seed/date 別 metrics、4 registered contrasts、matched latency protocol が揃い、deviation がある場合は自動昇格せず手動監査すること。
6. aggregate/seed/date 別 primary metric、contrast、Holm 値、paired-unit 数、guardrail flag が、提出された row-level evidence から完全に再計算できること。完全な実験の guardrail 不合格は証拠として受理するが、成功主張には使わない。
7. fail-closed checker が `admissible: true` を返した後、結果の方向を反映した JSON pointer 付き ledger 更新と原稿再生成を行うこと。

読み取り専用の admission check は次で実行します。

```bash
make check-evidence RUN_MANIFEST=/path/to/run_manifest.json EVALUATION=/path/to/evaluation.json PAIRED_OBSERVATIONS=/path/to/paired_observations.jsonl
```

checker は immutable hash link、code revision、source/teacher identity、split/seeds、tree-lock timing、metric completeness、再計算した aggregate/seed/date 別 primary metric と bootstrap/Holm contrast、再計算した quality/diversity/latency guardrail flag を検証します。証拠の妥当性と効果主張の成否は分離します。ledger や原稿は自動編集しません。

## 投稿時の匿名性

この開発リポジトリは所有者を特定できるため、匿名査読原稿から直接リンクできません。証跡 gate 通過後に匿名 snapshot を作ります。所属、謝辞、公開 artifact URL は review/camera-ready の移行まで非表示または placeholder のままです。

原稿ソースの著者名は、名前を NAOYUKI、家族名を TOMITA として **NAOYUKI TOMITA** と記録しています。review build は ACM の `anonymous=true` を維持するため `paper/main.pdf` では実名が非表示ですが、`paper/main-author.pdf` では表示されます。所属は未提示のため推測せず省略しています。
