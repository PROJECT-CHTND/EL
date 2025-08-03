# EL Pipeline PoC – Evaluation Guide

このドキュメントでは、CLI (`scripts/pipeline_cli.py`) を用いた暗黙知抽出パイプラインの検証手順と、評価指標・サンプルデータセット例を示します。

---
## 1. 目的
1. 抽出された知識グラフの **網羅性 / 正確性** を定性的に確認する
2. 生成された質問の **specificity / tacit_power** が十分に高いかを評価する
3. 推論コスト (token 消費量) と処理時間を把握する

---
## 2. 必要環境
- OpenAI API キー (gpt-4o)
- Neo4j (Aura Free でも可) ※オフラインなら `AUTO_INGEST_NEO4J=0` で実行可
- Python 3.11 + 依存ライブラリ

```bash
make install  # .venv 作成 & 依存インストール
```

---
## 3. ゴールドデータの作成

### 3.1 知識グラフ用 (Entities / Relations)

テンプレートは `data/templates/entities.csv`, `relations.csv` をコピーして使用。

```
entities.csv
id,label,type,判定,備考

relations.csv
source_id,target_id,type,判定,備考
```

* **id** は一意文字列 (見出し語英語など)
* **判定** はシステム抽出後に y/n を付けて正確性を測定
* 1 記事あたり 20〜30 エンティティ / 30〜50 リレーションを目安

### 3.2 質問品質アノテーション

`questions_annotated.jsonl` (1 行 1 JSON)

```jsonc
{
  "article_id": "ai_01",
  "slot_name": "date_of_birth",
  "question": "When were you born?",
  "clarity": 4,        // 1–5
  "specificity": 5,
  "tacit_power": 4,
  "overall": 4
}
```

3 名以上で採点し平均を使用。`overall>=4` を「有効」と定義し Precision / Recall を算出。

テンプレート生成スクリプト例:

```bash
python scripts/init_dataset.py ai_01 path/to/article.txt  # TODO: 実装
```

---
## 3. CLI の使い方
```bash
# 基本実行
python scripts/pipeline_cli.py data/sample_ai.txt --focus "AI" --temp 0.3
```

出力:
```
===== Stage01: Hierarchical Context =====
{ ... three-layer summaries ... }

===== Stage02: Knowledge Extraction =====
{ "entities": [...], "relations": [...] }

===== Stage04: Slot Discovery =====
- date_of_birth : Add date of birth info
...

===== Stage06/07: Question Generation & QA =====
Q: When were you born? (spec=0.80, tacit=0.92)
...
```

---
## 4. 評価指標
| 観点 | 指標 | 取得方法 |
|------|------|----------|
| KG 網羅性 | 重要エンティティ抜け漏れ数 | 人手アノテーション (ゴールドリスト) と比較 |
| KG 正確性 | 誤エッジ率 (%) | Neo4j Browser で目視確認 / 手動カウント |
| 質問品質 | specificity, tacit_power | LLM 自動スコア (≥0.7) と人手 5 段階評価 |
| コスト | input/output tokens | `logs/llm_calls.jsonl` 集計 |
| レイテンシ | パイプライン総処理時間 | CLI 実行時間 / Prometheus `request_duration_seconds` |

---
## 5. サンプル評価データ
`data/` ディレクトリに以下を配置するとすぐに検証可能です。

| ファイル | 内容 | 用途 |
|----------|------|------|
| `sample_ai.txt` | AI 技術の歴史・応用についての記事 (~1500 words) | 技術ドメインでの抽出テスト |
| `sample_health.txt` | 生活習慣病に関するガイド (~1200 words) | 医療ドメインでの抽出テスト |
| `sample_history.txt` | 中世ヨーロッパ史の出来事まとめ (~1000 words) | 人文ドメインでの抽出テスト |

> *ライセンスフリーの Wikipedia コンテンツを貼り付けると手軽です。*

---
## 6. 評価手順例
1. 各サンプルテキストを CLI に投入し、Neo4j Browser でグラフを確認
2. 抽出されたエンティティ/リレーションが記事の主要概念をカバーしているかをチェック (網羅性)
3. 誤リンク・誤属性が無いかをチェック (正確性)
4. 生成質問を 3 名以上の評価者が 1–5 の Likert スケールで採点し平均を計算
5. コストと時間が要件内か確認 (例:1,000 words → ≤5s, ≤5k tokens)

---
## 7. レポートテンプレート
```
### Article: sample_ai.txt
- Nodes: 45, Edges: 52
- Missing key entities: 2 / 20 (10%)
- Wrong edges: 1
- Avg question specificity (human): 4.3 / 5
- Avg tacit power (human): 4.1 / 5
- Tokens in/out: 4,850 / 1,120
- Pipeline latency: 4.7s
- Notes: ...
```

---
## 9. 評価パイプライン自動化

`make eval-offline` などのターゲットを今後追加予定。処理内容:
1. `pipeline_cli.py` で質問生成 → `questions_raw.jsonl`
2. 手動アノテーション後に `eval_offline.py` を実行
3. 出力: `eval_metrics.csv` (Precision, Recall, F1, コスト, レイテンシ)

オンライン評価 (`eval_online.py`) では回答 JSONL と KG Δ を入力し `SlotFillRate` 等を自動算出。

---
これに従い数記事で評価を行うことで、パイプライン設計の妥当性と改善ポイントを定量・定性の両面で把握できます。 