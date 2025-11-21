## SQLite セッションストア移行ガイド（M1b）

EL の M1b では、Discord セッション状態を SQLite で永続化するために
`sessions` / `messages` / `slots` テーブルを使用します。

新規環境では、`SqliteSessionRepository.init()` が自動でスキーマを作成するため、
**マイグレーション作業は不要**です。

- 実装: `agent/stores/sqlite_store.py` の `SqliteSessionRepository.init()`
- デフォルトパス: `EL_SQLITE_PATH`（未指定時は `./data/el_sessions.db`）

```python
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        thread_id INTEGER,
        topic TEXT NOT NULL,
        goal_kind TEXT NOT NULL,
        language TEXT,
        created_at TEXT NOT NULL
    );
    """
)
...
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS slots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        type TEXT,
        importance REAL,
        filled_ratio REAL,
        last_filled_ts REAL,
        value TEXT,
        evidence_ids_json TEXT,
        source_kind TEXT,
        UNIQUE(session_id, name),
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    """
)
```

---

### 1. 既存 DB がない/捨ててよい場合

もっともシンプルなパスです。古いテーブル定義やデータが不要な場合は、
DB ファイルを削除してから起動してください。

```bash
rm data/el_sessions.db  # または EL_SQLITE_PATH で指定したパス
python nani_bot.py      # 起動時に自動で sessions/messages/slots が作成される
```

---

### 2. 既存セッション DB を残したい場合（手動 ALTER 例）

すでに `sessions` / `messages` テーブルを持つ `el_sessions.db` があり、
そのデータを引き継いで M1b スキーマに移行したい場合は、
以下のような手順で ALTER を行ってください。

#### 2.1 移行対象 DB のバックアップ

```bash
cp data/el_sessions.db data/el_sessions.backup.$(date +%Y%m%d%H%M).db
```

#### 2.2 必要なカラム/テーブルを追加

```bash
sqlite3 data/el_sessions.db
```

`sqlite>` プロンプトで、次の SQL を順に実行します。

```sql
-- まだなければ thread_id / language カラムを追加
ALTER TABLE sessions ADD COLUMN thread_id INTEGER;
ALTER TABLE sessions ADD COLUMN language TEXT;

-- M1b で使用する slots テーブルを作成（存在しない場合のみ）
CREATE TABLE IF NOT EXISTS slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    type TEXT,
    importance REAL,
    filled_ratio REAL,
    last_filled_ts REAL,
    value TEXT,
    evidence_ids_json TEXT,
    source_kind TEXT,
    UNIQUE(session_id, name),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_slots_session ON slots(session_id, name);
```

その後 `.quit` で sqlite3 を終了します。

```sql
.quit
```

#### 2.3 動作確認

1. EL を起動し、Discord でセッションを開始する。
2. いくつかのターンを進めたあとでプロセスを再起動する。
3. 同じスレッドでメッセージを送信し、`last_question` / スロットが復元されることを確認する。

テストスイートからは、以下のテストが `sessions` / `messages` / `slots` の
基本的な CRUD とスロット永続化をカバーします。

- `tests/test_sqlite_store.py::test_sqlite_store_crud_and_slots`

---

### 3. 環境変数によるパス指定

運用環境ごとに DB パスを分けたい場合は、`EL_SQLITE_PATH` を設定してください。

```bash
export EL_SQLITE_PATH=/var/lib/el/el_sessions.db
python nani_bot.py
```

この場合も、**新規作成時は自動で M1b スキーマが作成**され、
既存 DB を引き継ぐ場合は本ドキュメントの ALTER 手順をそのパスに対して実施します。


