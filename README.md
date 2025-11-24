## EL (Eager Learner) / 好奇心駆動インタビューエージェント

EL は、**Discord 上で動作する思考パートナーBot** と  
**暗黙知抽出・VoI（情報価値）ベースのエージェント群** をまとめたリポジトリです。

- 対話を通じてユーザーの考えを深掘りする Discord Bot「EL」
- テキストから KG（知識グラフ）を抽出する FastAPI（`agent/`）
- VoI/Slot/Strategist を備えた次世代エージェント基盤（`el-agent/`）

詳細なアーキテクチャとロードマップは `docs/CURIOUS_AGENT_FINAL.md` を参照してください。

---

## 1. まず試すなら：Discord Bot EL

### 1.1 必要環境

- Python 3.11 推奨（3.8 以上で動作想定）
- Discord Bot トークン
- OpenAI API キー

### 1.2 セットアップ

```bash
git clone https://github.com/PROJECT-CHTND/EL.git
cd EL

pip install -r requirements.txt
```

ルートに `.env` を作成し、最低限の環境変数を設定します。

```bash
DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
OPENAI_MODEL="gpt-4o"

# セッション永続化（SQLite）
EL_SQLITE_PATH="./data/el_sessions.db"

# トレース / WAL ログ
EL_TRACE=1
EL_TRACE_DIR="./logs/wal"

# メトリクス公開ポート（Prometheus）
METRICS_PORT=8000
```

### 1.3 起動と使い方

```bash
python nani_bot.py
```

- コンソールに `🧠 EL has started!` が出れば起動成功
- Discord の任意チャンネルで:

```text
!explore 最近感動したこと
```

Bot が専用スレッドを作成し、そのトピックに対して**日本語または英語で探求的な質問**を投げかけます。

利用可能な主なコマンド:

- `!explore [トピック]`  
  新しいセッションを開始（言語は自動判定）
- `!reflect`  
  これまでの対話から「主な発見」「深まった理解」「今後の問い」を要約
- `!finish`  
  対話ログ（Markdown）と RAG 用 JSONL を生成してスレッドに添付

---

## 2. API として使う：Implicit Knowledge Extraction Agent（`agent/`）

テキストからエンティティ・リレーション・スロット候補を抽出する FastAPI アプリです。

### 2.1 起動

```bash
pip install -r requirements.txt

uvicorn agent.api.main:app --reload
```

- OpenAPI: `http://localhost:8000/docs`
- 主なエンドポイント:

| Method | Path                | 説明                                          |
|--------|---------------------|-----------------------------------------------|
| POST   | `/extract`          | KG 抽出（Stage②）→ `KGPayload` を返す       |
| POST   | `/stream_pipeline`  | Stage01–04 の進捗を SSE でストリーミング    |
| POST   | `/kg/submit`        | KG 事実の登録                                 |
| POST   | `/kg/approve`       | KG 事実の承認                                 |
| GET    | `/metrics`          | Prometheus 形式のメトリクスを返却            |

環境変数（例）:

```bash
OPENAI_API_KEY="sk-..."
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password"
REDIS_URL="redis://localhost:6379"          # 任意
EL_EVAL_WEIGHTS="/absolute/path/to/config/weights/weights.json"  # 任意
EL_PROMPT_VARIANTS_DIR="/absolute/path/to/prompts"               # 任意
```

---

## 3. VoI/Slot ベースのエージェント基盤：`el-agent/`

`el-agent/` 以下には、VoI（情報価値）と Slot/Strategist を用いた次世代エージェントのコア実装があります。

- `core/strategist.py` : `ask/search/none` を選択する Strategist
- `core/knowledge_integrator.py` : BM25 / ベクトル検索の統合
- `monitoring/metrics.py` : Prometheus メトリクス

詳細な起動方法は `el-agent/README.md` を参照してください（Poetry ベース）。

### 3.1 VoI / 停止しきい値の設定（M2 準備）

Strategist の VoI/停止しきい値は、環境変数で調整できます。

- `EL_VOI_TAU_STOP`  
  `ask/search/none` を切り替えるための停止しきい値（既定: `0.08`）

```bash
export EL_VOI_TAU_STOP=0.05  # しきい値を下げて、より積極的に質問を続ける
```

---

## 4. 可観測性とメトリクス

- Discord Bot 起動時、`METRICS_PORT` で Prometheus 互換メトリクスを公開
- `ops/prometheus.yml` / `ops/grafana/` により、ローカルで

```bash
docker compose up -d prometheus loki promtail grafana
```

を実行すると、Grafana の「EL Agent Overview」ダッシュボードから

- Turn latency (p50/p90/p99)
- Slot coverage
- QCheck duplicate rate
- WAL ログ（Loki）

などを閲覧できます。

---

## 5. テストとローカル CI

### 5.1 pytest

ルートのテストは pytest で実行できます。

```bash
python -m pytest -q
```

### 5.2 Makefile による簡易 CI

```bash
make install   # .venv 作成 + 依存インストール
make ci        # pytest -q + mypy agent
```

Python バージョンを変えたい場合:

```bash
make PYTHON=python3.10 install
```

---

## 6. 参考ドキュメント

- `docs/CURIOUS_AGENT_FINAL.md`  
  好奇心駆動インタビューエージェントの最終設計書（M1〜M3・R4）
- `docs/R4_RESEARCH_TRACK.md`  
  研究トラック（方策最適化/VoI チューニング等）のメモ

これらを読みつつ、まずは **「1. Discord Bot EL」** から動かしてみると、
リポジトリ全体の目的と挙動をつかみやすくなります。