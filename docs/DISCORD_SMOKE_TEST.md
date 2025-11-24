## Discord スモークテスト手順（M1b）

本ドキュメントは、M1b 好奇心エージェントの代表的な Discord スモークテスト手順をまとめたものです。
新しい環境構築時やリリース前の簡易確認に利用します。

---

### 1. 前提

- `python nani_bot.py` で Discord Bot が起動済み（`🧠 EL has started!` が表示されている）
- `METRICS_PORT` が設定されており、`curl localhost:$METRICS_PORT/metrics` でメトリクスが取得できる
- Prometheus / Loki / Promtail / Grafana が `docker compose up -d prometheus loki promtail grafana` で起動済み

---

### 2. ポストモーテム・クリティカルスロットの進行確認

目的: ポストモーテムセッションで、主要5スロット
`summary → impact → detection_ttd → timeline → CAPA`
が自然な順で進行することを確認します。

1. Discord で任意のテキストチャンネルを開き、以下を送信:

   ```text
   !explore 昨日の決済システム障害について振り返りたい
   ```

2. Bot が専用スレッドを作成し、最初の質問を投げかけます。
   - 最初の質問は「概要(summary)」を聞く内容になっていること

3. 回答例（自由に書き換え可）:

   ```text
   昨日14:30頃、決済APIでタイムアウトが発生し、約2時間決済できなくなりました。
   ```

4. 以降、回答を続けながら、質問のフォーカスが以下の順番で変化することを確認します。
   - 影響範囲（impact）
   - 検知方法・TTD（detection_ttd）
   - 主要イベントの時系列（timeline）
   - 是正・予防策（CAPA）

5. 主要スロットが埋まったタイミングで、Bot が「ポストモーテムの準備完了」相当のメッセージを返すことを確認します。

---

### 3. プロセス再起動後のセッション継続

目的: SQLite 永続化を利用して、プロセス再起動後も同じスレッドでセッションが継続されることを確認します。

1. 上記 2. の途中（例: impact まで進んだ状態）で、Bot プロセスを手動で停止します。
2. 同じマシンで再度 `python nani_bot.py` を実行し、Bot を起動します。
3. 先ほどのスレッドに戻り、新しいメッセージを投稿します。
4. 期待される動作:
   - Bot が `SqliteSessionRepository` からセッションを復元し、
     直前の `last_question` とスロット状態を再利用して次の質問を生成する。
   - 途中まで埋まっているスロット（たとえば `summary` / `impact`）が二重に質問されない。

---

### 4. メトリクスとダッシュボードの反映確認

目的: 対話中のメトリクスが Prometheus / Grafana に反映されていることを確認します。

1. 対話を数ターン進めた状態で、Prometheus UI または `curl` から以下のメトリクスを確認:
   - `turn_latency_seconds_bucket`
   - `slot_coverage`
   - `slot_duplicate_rate{pipeline_stage="stage07_qcheck"}`

2. Grafana で「EL Agent Overview」ダッシュボードを開き、以下を確認:
   - Turn latency の p50/p90/p99 グラフに最近のトラフィックが反映されている
   - Slot coverage / QCheck duplicate rate のグラフが描画されている
   - WAL logs パネルに最新の `logs/wal/*.log` がストリームされている

---

### 5. 失敗時のチェックポイント

スモークテスト中に問題があった場合、以下を順に確認します。

- Bot が起動しているか（`DISCORD_BOT_TOKEN` が正しいか）
- `EL_SQLITE_PATH` のパスに書き込み権限があるか
- `METRICS_PORT` で /metrics が開いているか
- Promtail / Loki / Grafana が `ops/` 配下の設定で正しく起動しているか

これらを満たしていれば、M1b における「Discord 経由の基本動作 + 永続化 + 可観測性」のスモークが完了している状態となります。


