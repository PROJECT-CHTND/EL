# 暗黙知抽出エージェント実装計画

本ドキュメントは、Discord／Teams Bot 等の UI 層を除き、LLM API と知識グラフを中心とした暗黙知抽出エージェントの E2E パイプライン設計および運用手順をまとめたものである。

---
## 1. アーキテクチャ全体像

| 層 | 主要コンポーネント | 役割 |
|----|------------------|------|
| API 層 | FastAPI | `/extract`, `/generate_q`, `/stream_pipeline` を提供し、クライアントとパイプラインの境界となる |
| オーケストレーション層 | `AsyncPipelineRunner` | 各ステージの並列実行・リトライ・ログ収集を担当 |
| LLM Wrapper | `OpenAIClient (gpt-4o)` | function-calling、JSON バリデーション、logprobs 取得、リトライ制御 |
| KG ストア | Neo4j / Memgraph | 知識グラフを格納、confidence <0.7 のノードに `needs_review` フラグ |
| スロット管理 | Redis (Hash) | 未充足スロットの登録・更新、GapAnalyzer が優先度を算出 |
| ログ & メトリクス | Prometheus / Grafana / S3 | トークン量・処理時間・エラー率を可視化

---
## 2. ディレクトリ構成例
```text
/agent
  ├── api/              # FastAPI ルータ
  ├── pipeline/         # 各ステージ実装
  │   ├── stage01_context.py
  │   ├── stage02_extract.py
  │   ├── stage03_merge.py
  │   ├── stage04_slots.py
  │   ├── stage05_gap.py
  │   ├── stage06_qgen.py
  │   └── stage07_qcheck.py
  ├── llm/              # gpt-4o Wrapper & schemas
  ├── kg/               # GraphDB 接続・KGDeltaBuilder
  ├── slots/            # SlotRegistry & GapAnalyzer
  ├── models/           # pydantic schemas
  ├── utils/            # 共通ユーティリティ
  └── tests/            # pytest
```

---
## 3. ステージ別実装メモ

### Stage① 階層コンテキスト生成
* 入力全文から 3 層要約（RAG キー／中位概要／グローバル概要）を生成。
* 結果を Redis Stream に publish し、後続ステージが subscribe。

### Stage② 段階的知識抽出（Explicit → Implicit → Exception → Nuance）
* 共通 function: `save_kv`。
* サブステージごとに temperature と focus プロンプトを変更。
* 抽出結果 JSON は pydantic で検証後 Neo4j に MERGE 挿入。

### Stage③ KG マージ & 信頼度推定
* 同一ノード判定: Jaro-Winkler >0.9 または embedding cosine >0.8。
* `confidence = 平均(logprob) × stage_weight`。
* 0.7 未満のノード／属性は `needs_review` を付与。

### Stage④ 動的スロット発見
* Input: current KG, topic meta。
* function `propose_slots` で最大 3 件のスロット提案。

### Stage⑤ ギャップ分析（Python）
* SlotRegistry と KG を突合し、優先度 `importance × (1 - filled_ratio) × recency_boost` を算出。

### Stage⑥ 質問生成
* function `generate_questions`。
* `strategy_map` は YAML で管理し、slot_type ごとにテンプレートを定義。

### Stage⑦ 質問 QA
* function `return_validated_questions`。
* specificity・tacit_power ≥0.7 で accept、未満なら再生成。

---
## 4. エラー／矛盾処理フロー
1. **JSON バリデーション失敗** → 同一メッセージ再送 (最大 2 回, temp=0)
2. **KG 矛盾** (cosine <0.8) → function `resolve_conflict` で自動解決 or ユーザー確認メッセージ生成
3. **Hallucination 判定**: `confidence <0.3 && not in text window` → 破棄しログ記録

---
## 5. ストリーミング設計
FastAPI `StreamingResponse` + async generator
```python
yield {"stage":"rough_kg","progress":40,"kg":...}
yield {"stage":"missing_slots","progress":70,"slots":...}
yield {"stage":"complete","progress":100,"follow_up":...,"kg_delta":...}
```
フロントエンドは EventSource で受信しリアルタイム更新。

---
## 6. テスト戦略
* **単体テスト**: 各ステージ JSON schema 準拠を pytest で検証。
* **結合テスト**: mock LLM で全文 → 質問生成までを再現。
* **回帰テスト**: 過去データ 100 本を nightly で再実行し diff==0 を確認。
* **性能テスト**: pytest-benchmark で tokens/秒 を測定。

---
## 7. 今後の拡張ポイント
1. Temporal KG で因果・時間属性を付与。
2. ベクトル DB (Qdrant) 連携で過去セッション横断検索。
3. “suggest_action” ステージで意思決定支援へ拡張。

---
## 8. マイルストーン例
| 週 | 内容 | Deliverable |
|----|------|-------------|
| 1  | API 骨格 + LLM Wrapper | `/extract` v0 (Stage②のみ) |
| 2  | KG 統合 + 信頼度計算 | `/extract` フル |
| 3  | SlotRegistry + 質問生成 | `/generate_q` v1 |
| 4  | ストリーミング & UI モック | SSE デモ |
| 5  | 負荷試験・コスト試算 | レポート |
| 6  | ドキュメント & v1 リリース | README, OpenAPI |

---
## 9. セキュリティ & コンプライアンス
### 9-1. API 認証
* JWT (Auth0) による保護、API Gateway で IP 制限。

### 9-2. データ暗号化
* Neo4j encrypted mode + volume encryption。
* Redis TLS + AUTH。
* S3 ログは SSE-S3 (AES-256)。

### 9-3. PII マスキング
* LLM 送信前に Presidio でフィルタリング。
* 返却 JSON に PII 判定タグを付与。

### 9-4. 監査ログ
* すべての LLM 呼び出しを JSONL で保存、監査専用 IAM ロールで閲覧。

---
## 10. CI / CD パイプライン
* GitHub Actionsワークフロー
  1. **test**: pytest / mypy / schema check
  2. **build**: Docker multi-stage build
  3. **scan**: Trivy & secrets-scan
  4. **deploy-stg**: ArgoCD + Helm
  5. **e2e-test**: mock LLM でパイプライン全通し
  6. **deploy-prod**: manual approval

* バージョン管理: SemVer, Docker image `ghcr.io/org/implicit-agent:{git-sha}`

---
## 11. インフラ & デプロイ
| コンポーネント | サービス | HA |
|---------------|----------|----|
| FastAPI | AWS ECS Fargate | 2 AZ |
| Neo4j | Neo4j Aura | Daily backup |
| Redis | AWS ElastiCache | Multi-AZ |
| Vector DB | Qdrant Cloud | Sharded |
| ログ | AWS S3 | Versioning |
| モニタリング | Prometheus + Grafana Cloud | — |

---
## 12. 運用監視 & コスト管理
* **メトリクス**: tokens_in/out, processing_ms, gpt4o_calls, retry_count。
* **アラート**: 95th processing_ms > 8s, 月間トークン > 3M。
* **ダッシュボード**: Stage別処理時間ヒートマップ、confidence 分布。

---
## 13. 簡易リファレンス実装スニペット
```python
async def extract_knowledge(text: str, focus: str, temp: float) -> KGPayload:
    prompt = [
        {"role": "system", "content": "You are a structured-data extractor."},
        {"role": "user", "content": f"## Input\n{text}\n## Required schema\n{\"entities\":[],\"relations\":[]}"}
    ]
    resp = await openai_client.call(
        model="gpt-4o",
        temperature=temp,
        messages=prompt,
        functions=[schemas.save_kv],
        function_call={"name": "save_kv"},
        logprobs=True
    )
    payload = KGPayload.model_validate(resp.function_call.arguments)
    return payload.with_confidence(resp.logprobs)
```

---
## 14. データライフサイクル方針
* KG: 1 年保持後にアーカイブ。
* LLM ログ: 90 日保持、GDPR 準拠で削除。
* user_id 指定の削除 API を用意し、Neo4j + S3 + Redis から一括削除。

---
## 15. リスク & 対応
| リスク | 対応策 |
|--------|--------|
| LLM 出力変動による JSON 崩壊 | リトライ + temp=0、最後にスキーマ fallback |
| コスト急増 | 月次トークン上限設定、超過時に STOP フラグ |
| KG サイズ肥大 | スナップショット→Parquet で DataLake 保存 |
| モデル更新 (gpt-4o-v2) | Canary Pipeline で差分検証 |

---
### ◆ まとめ
* すべての知識抽出・生成タスクを **gpt-4o**＋function-calling に統一。
* Neo4j＋Redis のシンプル構成で、高速な KG とスロット管理を実現。
* 非同期並列実行 & SSE により **5–6 秒** でフルパイプライン応答。
* 信頼度・矛盾処理を組み込み、品質と安全性を両立。 