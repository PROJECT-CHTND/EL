# EL (Eager Learner) v2

**好奇心駆動インタビューエージェント** - LLMネイティブな対話システム

ELは、GPT-5.2をベースにした共感的なインタビューエージェントです。
ユーザーとの対話を通じて洞察を引き出し、知識グラフに保存します。

## 特徴

- 🧠 **LLMネイティブアーキテクチャ**: GPT-5.2のFunction Callingを活用
- 💭 **共感的な対話**: 傾聴と深掘りを重視したインタビュースタイル
- 🔍 **動的ドメイン認識**: 会話内容から自動的にドメインを判断
- 📊 **知識グラフ統合**: Neo4jで洞察を永続化・検索
- 🎨 **モダンなWeb UI**: リアルタイムチャットインターフェース

## アーキテクチャ

```
EL/
├── packages/
│   ├── core/                 # エージェントコア
│   │   └── src/el_core/
│   │       ├── agent.py      # メインエージェント
│   │       ├── tools.py      # KG検索・保存ツール
│   │       ├── prompts.py    # System Prompt
│   │       ├── schemas.py    # Pydanticスキーマ
│   │       ├── llm/          # LLMクライアント
│   │       └── stores/       # KGストア
│   │
│   ├── api/                  # FastAPI バックエンド
│   │   └── src/el_api/
│   │       └── main.py       # APIエンドポイント
│   │
│   └── web/                  # Web UI
│       └── index.html        # チャットインターフェース
│
├── pyproject.toml            # ワークスペース設定
├── docker-compose.yml
└── Makefile
```

## クイックスタート

### 1. 環境設定

```bash
# リポジトリをクローン
git clone https://github.com/PROJECT-CHTND/EL.git
cd EL

# 環境変数を設定
cp env.example .env
# .envを編集してOpenAI APIキーを設定
```

### 2. 依存関係のインストール

```bash
# uvを使用（推奨）
make install

# または手動で
pip install uv
uv sync
```

### 3. 起動

```bash
# APIサーバーを起動
make run

# 開発モード（ホットリロード）
make run-dev
```

### 4. ブラウザでアクセス

```
http://localhost:8000
```

## API エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| `POST` | `/api/sessions` | 新しいセッションを開始 |
| `POST` | `/api/sessions/{id}/messages` | メッセージを送信 |
| `GET` | `/api/sessions/{id}` | セッション情報を取得 |
| `DELETE` | `/api/sessions/{id}` | セッションを終了 |
| `WS` | `/ws/{session_id}` | WebSocketリアルタイムチャット |

## Docker での起動

Neo4j知識グラフを含むフルスタックを起動：

```bash
# サービスを起動
make docker-up

# ログを確認
make docker-logs

# 停止
make docker-down
```

## 設定

### 必須環境変数

| 変数名 | 説明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI APIキー |

### オプション環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `OPENAI_MODEL` | `gpt-5.2` | 使用するモデル |
| `PORT` | `8000` | APIサーバーポート |
| `NEO4J_URI` | - | Neo4j接続URI |
| `NEO4J_USER` | `neo4j` | Neo4jユーザー |
| `NEO4J_PASSWORD` | - | Neo4jパスワード |
| `LOG_LEVEL` | `INFO` | ログレベル |

## 開発

```bash
# 開発用依存関係をインストール
make dev

# リンター実行
make lint

# フォーマット
make format

# テスト実行
make test
```

## ドメイン

ELは以下のドメインを自動認識します：

| ドメイン | 例 |
|---------|-----|
| `daily_work` | 業務日報、タスク進捗 |
| `recipe` | 料理、レシピ |
| `postmortem` | 障害振り返り |
| `creative` | 創作活動、アイデア |
| `general` | その他一般 |

## ライセンス

MIT License
