## V2 機能概要（現状）

本ドキュメントは、EL の v2（仮説検証型アーキテクチャ）の「現状実装済み機能」をコンパクトに説明します。詳細な設計意図は `IMPLEMENTATION_PLAN.md`、運用手順は `docs/OPERATIONS.md` を参照してください。

---

### 目的と全体像

- 直線型の v1 パイプラインから、反復的な「仮説→知識統合→証拠抽出→評価→次アクション」へ移行。
- 対話の深さ・コスト・外部知識を踏まえ、次の質問や検索を選択して精度と効率を両立。

---

### 主要コンポーネント（実装済み）

- Orchestrator（`el-agent/src/el_agent/core/orchestrator.py`）
  - 推論ループのウォールログ（WAL）を `logs/wal/YYYY-MM-DD.log` へ出力。
  - WAL には PII マスキング（メール・電話・住所のトークン化、`user_id_hash`）を適用。

- Strategist（`el-agent/src/el_agent/core/strategist.py`）
  - 仮説の不確実性（エントロピー）と行動コストを用いて Value-of-Information を計算。
  - `ask | search | none` のいずれかを返す停止しきい値（τ）つき戦略選択を実装。

- Knowledge Integrator（`el-agent/src/el_agent/core/knowledge_integrator.py`）
  - Elasticsearch（BM25）と Qdrant（ベクトル近傍）の並列検索を実行し、RRF（Reciprocal Rank Fusion）で統合。
  - ルールベースの軽量リランキング、文単位抽出（クエリ重なり・簡易コサイン類似度）、`Evidence` 生成を提供。
  - Qdrant の探索強度 `hnsw_ef` を `config/qdrant_ef.txt` からランタイム読込。

- Evaluator / Confidence（`el-agent/src/el_agent/core/evaluator.py`）
  - 特徴量（`cosine`, `source_trust`, `recency`, `logic_ok`, `redundancy` など）から信頼度寄与を算出。
  - 仮説の信念 `belief` をロジット空間で更新し、信念区間（CI）を縮小。
  - 事前較正済みの重みを `config/weights/weights.json` または `EL_EVAL_WEIGHTS` で外部読込可能。

- LLM / Prompts（`el-agent/src/el_agent/llm/`）
  - 役割ごとの機能を JSON 厳守で実装：仮説候補生成・証拠抽出・質問生成・文書合成・質問ストラテジスト・QA リファイン。
  - プロンプトは `EL_PROMPT_VARIANTS_DIR` など外部ディレクトリから差し替え可能（内蔵フォールバックあり）。

- Stores / Retrieval
  - `ESClient`（`el-agent/src/el_agent/retrieval/es_client.py`）: BM25 検索。
  - `QdrantStore`（`el-agent/src/el_agent/stores/qdrant_store.py`）: コレクション作成、バッチ Upsert、EF 設定、検索。
  - `Neo4jStore`（`el-agent/src/el_agent/stores/neo4j_store.py`）: 最小限のノード登録 API。
  - `RedisStore`（`el-agent/src/el_agent/stores/redis_store.py`）: JSON 文字列の get/set。

- Monitoring / Security
  - Prometheus メトリクス（`/metrics`）: レイテンシ、LLM コール、リトリーバル段階別カウント、オープン仮説数。
  - JWT 検証（Auth0 想定、`RS256`）。エンドポイントでの `Depends(verify_jwt)` 適用。
  - PII マスキングユーティリティ（`el-agent/src/el_agent/utils/pii.py`）。

---

### 推論ループの流れ（概要）

1) Strategist が仮説の不確実性・コストからアクションを選択（`ask|search|none`）

2) Knowledge Integrator が必要に応じて BM25 + ベクトル検索 → RRF → リランキング → 文抽出

3) LLM で証拠抽出（構造化 JSON）→ `Evidence` へ整形

4) Evaluator が `belief` を更新（ロジット加算 + CI 縮小）

5) 収束または停止条件を満たすまで繰り返し（WAL に追記）

---

### API（v2 関連）

- `POST /extract`（JWT 必須）: v1 パイプラインの抽出（Stage②）を実行して `KGPayload` を返却。
- `POST /stream_pipeline`（JWT, SSE）: context → extract → slots の進捗をストリーム配信。
- `POST /kg/submit` / `POST /kg/approve`（JWT）: KG 事実の登録・承認。
- `GET /metrics`: Prometheus メトリクス。

SSE 出力例（概略）:

```json
{"stage":"context","progress":20,"context":{...}}
{"stage":"extract","progress":60,"kg":{...}}
{"stage":"slots","progress":80,"slots":[...]}
{"stage":"complete","progress":100}
```

---

### 設定・環境変数（抜粋）

```bash
OPENAI_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
REDIS_URL=redis://localhost:6379
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_AUDIENCE=https://api.example.com
PII_SALT=random_salt
EL_EVAL_WEIGHTS=/absolute/path/to/config/weights/weights.json   # 任意
EL_PROMPT_VARIANTS_DIR=/absolute/path/to/prompts                # 任意（プロンプト外部化）
```

プロンプトは原則「外部ファイルから読み込む」運用を想定（ハードコード回避）。

---

### メトリクス（例）

- `request_latency_seconds{endpoint="..."}`: エンドポイント別レイテンシ（ヒストグラム）
- `llm_calls_total{kind="..."}`: LLM 呼び出し回数
- `retrieval_calls_total{stage="bm25|vector|rrf|rerank"}`: 検索段階別カウント
- `hypotheses_open`: オープンな仮説数（Gauge）

---

### 既知の制約（現状）

- Strategist/Knowledge Integrator は一部スタブ/ヒューリスティック実装（高速に動作するが、将来差し替え前提）。
- `Neo4jStore` は最小機能のみ。完全な KG スキーマやマージ戦略は別途実装が必要。
- LLM 出力の JSON 妥当性はプロンプトで担保しているが、堅牢性向上の余地あり。

---

### 今後の拡張（抜粋）

- クロスエンコーダによる本格的な再ランキング、埋め込み生成の導入
- Strategist の強化（ユーザ応答を踏まえた VoI 推定・停止ルールの洗練）
- KG マージ/承認フローの拡充、ドキュメント合成のテンプレート化・品質ゲート

---

### クイックスタート

1) 依存インストール

```bash
pip install -r requirements.txt
```

2) 環境変数を設定（上記を参照）

3) サーバ起動

```bash
uvicorn agent.api.main:app --reload
```

4) API 確認

- Swagger: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`


