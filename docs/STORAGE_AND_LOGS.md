## ストレージとログ運用ガイド（SQLite / WAL / logs）

### 1. SQLite セッションストア

- 実装: `agent/stores/sqlite_store.py`
- 既定パス: `EL_SQLITE_PATH`（未設定時は `./data/el_sessions.db`）
- PRAGMA:
  - `journal_mode=WAL`
  - `foreign_keys=ON`

#### 1.1 バックアップ方針

SQLite は単一ファイル DB のため、WAL モードでも以下の手順で安全にバックアップできます。

開発環境（ダウンタイム許容）:

```bash
# Bot / API を停止した状態で:
cp data/el_sessions.db data/el_sessions.backup.$(date +%Y%m%d%H%M).db
```

本番相当（オンラインバックアップ例）:

```bash
sqlite3 data/el_sessions.db ".backup 'data/el_sessions.backup.$(date +%Y%m%d%H%M).db'"
```

> WAL モードでは `-wal` / `-shm` ファイルも補助的に作成されますが、
> `.backup` コマンドを使うと一貫性のあるスナップショットが取得されます。

#### 1.2 ローテーション

- 推奨: 日次で `.backup.YYYYMMDDHHMM.db` を作成し、7〜30日分を保持
- ディスク使用量が問題になる場合は、週次フルバックアップ＋日次差分を外部ストレージに退避する運用も検討してください。

詳細なスキーマや移行手順は `docs/SQLITE_MIGRATION.md` を参照してください。

---

### 2. WAL ログ（アプリケーションログ: `logs/`）

`logs/` ディレクトリには、対話やパイプラインのイベントログが JSONL / テキスト形式で保存されます。

- 例:
  - `logs/wal/YYYY-MM-DD.log` … WAL（Write Ahead Log）形式のイベント
  - `logs/*.log` … ユーザごとの対話ログ（開発中）

#### 2.1 保持期間の目安

- 開発環境: 7〜14日
- ステージング/本番: 30〜90日

Promtail を経由して Loki に転送する場合、**ローカルの `logs/` は短めにローテーションしつつ、Loki 側で長期保持**する構成が推奨です。

例（cron / logrotate のイメージ）:

```bash
# 7日より古いログを削除
find logs -type f -name '*.log' -mtime +7 -delete
```

#### 2.2 Promtail との連携

`ops/promtail-config.yaml` で `logs/` をスクレイプ対象にしているため、
ローテーションポリシーは Promtail が追跡できる範囲で行ってください。

---

### 3. 推奨環境変数まとめ

`.env` / `env.sample` では、ストレージとログに関連する変数として以下を定義します。

```bash
EL_SQLITE_PATH="./data/el_sessions.db"
EL_TRACE=1
EL_TRACE_DIR="./logs/wal"
METRICS_PORT=8000
```

これらを設定し、Runbook に従ってバックアップとローテーションを行うことで、
M1b 以降の運用でデータ喪失やディスク逼迫のリスクを下げることができます。


