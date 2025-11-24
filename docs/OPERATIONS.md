## 運用 Runbook（M1b）

### 1. 起動手順

#### 1.1 Discord Bot（EL）

1. 依存インストール

   ```bash
   pip install -r requirements.txt
   ```

2. 必要な環境変数を設定（例）

   ```bash
   export DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN"
   export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   export OPENAI_MODEL="gpt-4o"
   # セッションストア
   export EL_SQLITE_PATH="./data/el_sessions.db"
   # トレース／ログ
   export EL_TRACE=1
   export EL_TRACE_DIR="./logs/wal"
   # メトリクス
   export METRICS_PORT=8000
   ```

3. Bot 起動

   ```bash
   python nani_bot.py
   ```

   - コンソールに `🧠 EL has started!` が表示されれば成功
   - `curl localhost:${METRICS_PORT}/metrics` で Prometheus メトリクスが取得できる

#### 1.2 可観測基盤（Prometheus / Loki / Promtail / Grafana）

1. Docker が利用可能な環境で以下を実行:

   ```bash
   docker compose up -d prometheus loki promtail grafana
   ```

2. Grafana へのアクセス（デフォルト）
   - URL: `http://localhost:3000`
   - ダッシュボード: 「EL Agent Overview」

---

### 2. セッション永続化と復元

- 実装: `agent/stores/sqlite_store.py`
- 既定パス: `EL_SQLITE_PATH`（未指定時は `./data/el_sessions.db`）
- PRAGMA:
  - `journal_mode=WAL`
  - `foreign_keys=ON`

#### 2.1 新規環境

新規環境では、`SqliteSessionRepository.init()` が自動でスキーマを作成します。

```bash
rm -f data/el_sessions.db
python nani_bot.py  # 起動時に sessions / messages / slots テーブルが自動作成される
```

#### 2.2 既存 DB の復元

バックアップされた `el_sessions.db` を復元する場合は、Bot を停止した状態で
対象パスに上書き配置します。

```bash
cp /backup/el_sessions.backup.db data/el_sessions.db
python nani_bot.py
```

起動後、既存スレッドでメッセージを送信すると、`get_session_by_thread` を通じて
セッションが復元され、`last_question` や SlotRegistry が再利用されることを確認します。

---

### 3. ダッシュボードの見方（EL Agent Overview）

`ops/grafana/dashboards/el-agent-overview.json` には、以下の代表的なパネルが含まれます。

- **Turn latency (p50/p90/p99)**  
  - メトリクス: `turn_latency_seconds_bucket`
  - 意味: Discord 1ターンあたりのエンド・ツー・エンドレイテンシ

- **Slot coverage by stage**  
  - メトリクス: `slot_coverage{pipeline_stage=...}`
  - 意味: ステージごとのスロット充足率（0〜1）

- **QCheck duplicate rate / duplicates total**  
  - メトリクス:
    - `slot_duplicate_rate{pipeline_stage="stage07_qcheck"}`
    - `slot_duplicates_total{pipeline_stage="stage07_qcheck"}`
  - 意味: 質問が重複としてフィルタされた比率と件数

- **WAL logs (Loki)**  
  - ソース: `logs/wal/*.log`
  - 意味: Orchestrator / SlotRegistry / QCheck などの主要イベント

---

### 4. アラートの意味と一次対応

Prometheus にアラートルールを設定している場合（例: `ops/alert_rules.yml`）、
代表的なアラートと対応は以下の通りです。

- **QCheckFailureRateHigh**
  - 条件: `failed_qcheck / total_qgen` が一定閾値（例: 15%）を超過
  - 対応:
    - `logs/wal/*` から `stage07_qcheck` の `qcheck_fail_reason` を確認
    - プロンプト／閾値（specificity / tacit_power）の見直し

- **DuplicateQuestionRateHigh**
  - 条件: `slot_duplicate_rate{pipeline_stage="stage07_qcheck"}` が高止まり
  - 対応:
    - `planners.yaml` のステップ定義や SlotRegistry の設計を確認し、
      同じ情報を繰り返し聞いていないかをレビュー

- **TurnLatencyHighP90 / HighP99**
  - 条件: `turn_latency_seconds` の p90 / p99 がしきい値超え
  - 対応:
    - LLM API レイテンシ、Elasticsearch / Qdrant / Neo4j の状態確認
    - コンテキスト長や並列度を一時的に抑制

- **ElAgentMetricsDown**
  - 条件: `up{job="el-agent"} == 0`
  - 対応:
    - `METRICS_PORT` 設定とポート競合を確認
    - `nani_bot.py` が起動しているか／例外で落ちていないかをログで確認

---

### 5. 日常運用チェックリスト

1. Discord で `!explore` → セッション開始できるか
2. ポストモーテム対話で主要スロットが自然な順で埋まるか
3. プロセス再起動後も同一スレッドで対話が継続できるか
4. Grafana「EL Agent Overview」で
   - Turn latency / Slot coverage / QCheck duplicate が更新されているか
5. 重大アラートが発火していないか（または発火時に一次対応ができるか）

この Runbook に従うことで、M1b の運用（起動・復元・監視・アラート対応）を
最低限の手順でカバーできます。


