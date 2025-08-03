# EL (Eager Learner) - A Bilingual Thinking Partner

---

## Implicit Knowledge Extraction Agent (API) – WIP

このリポジトリには現在、Discord Bot「EL」に加えて、**暗黙知抽出エージェント (FastAPI)** の実装が進行中です。

| 完了ステージ | 概要 |
|--------------|------|
| Stage① Context | 元テキストから 3 層要約 (rag_keys / mid / global) を生成し、任意で Redis Stream へ publish |
| Stage② Extraction | OpenAI Function-calling でエンティティ・リレーションを抽出 (`save_kv`) |
| Stage③ Merge | 抽出 KG を Neo4j にマージ、logprob から信頼度を算出 |
| Stage④ Slots | KG を分析し、未充足スロットを最大 3 件提案 |
| Stage⑤ Gap Analysis | SlotRegistry から優先度を計算 (`importance × (1 - filled) × recency`) |
| Stage⑥ Question Gen | スロットごとの質問をテンプレートベースで生成 |
| Stage⑦ Question QA | specificity / tacit_power ≥ 0.7 の質問のみ採用 |

### ディレクトリ構成 (抜粋)

```
agent/
  api/            # FastAPI エンドポイント (/extract など)
  pipeline/       # 上記 7 ステージ実装
  prompts/        # システム・テンプレートプロンプト
  llm/            # OpenAIClient ラッパー & function schemas
  kg/             # Neo4j クライアント
  slots/          # SlotRegistry & GapAnalysis
  models/         # pydantic モデル
tests/            # pytest によるユニットテスト
```

### エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| POST | `/extract` | Stage①–③ (現在は②まで) を実行し、`KGPayload` を返却 |

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

追加で以下の環境変数を設定してください (例 `.env`):

```
OPENAI_API_KEY="sk-..."
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password"
REDIS_URL="redis://localhost:6379"          # 任意
AUTO_INGEST_NEO4J="1"                      # 自動マージを有効化
```

### テスト

pytest を利用して各ステージのユニットテストを実行できます。

```bash
pytest -q
```

### ローカルで CI を再現する

GitHub Actions と同じチェック (pytest + mypy) を手元で走らせるには Makefile を用意しています。

```bash
# 依存インストール (仮想環境 .venv)
make install

# 型チェック + テスト実行
make ci
```

`python3.11` が無い場合は `make PYTHON=python3.10 install` のようにバージョンを指定できます。

### 運用 Runbook

詳細な運用手順・アラート設定は `docs/OPERATIONS.md` を参照してください。

### パイプラインをコマンドラインで試す (`pipeline_cli.py`)

以下のスクリプトで、FastAPI サーバを立てずに 7 ステージ・パイプラインを単体実行できます。

```bash
python scripts/pipeline_cli.py input.txt --focus "AI" --temp 0.3
```

- **`input.txt`** : 抽出対象の本文ファイル (UTF-8) を指定します。
- **`--focus`**    : (任意) 重点的に抽出したいキーワード。例 `--focus "生成AI"`。
- **`--temp`**     : (任意) OpenAI temperature。デフォルト 0.0。

環境変数 `.env` に設定した `OPENAI_API_KEY` が必須です。さらに
`NEO4J_*` や `REDIS_URL` を設定すると、ステージ③で自動マージ、ステージ①で
Redis Stream への publish が行われます。

実行例の出力順:
1. Stage01 コンテキスト3分割
2. Stage02 KG 抽出
3. Stage04 スロット提案
4. Stage06/07 質問生成 → QA 結果

---

### OpenAPI

サーバー起動後、`/docs` (Swagger UI) または `/openapi.json` でスキーマを確認できます。

---

## 概要

ELは、Discord上で対話を通じてユーザーの考えや洞察を深める手助けをするためのBotです。指定されたトピックについて、Botが知識に基づいた探求的な質問を投げかけ、ユーザーが自分の考えを整理し、新しい視点を発見することをサポートします。

**日本語と英語の両方に対応しています。**

## 主な機能

-   **バイリンガル対話**: `!explore` コマンドで指定したトピックの言語（日本語または英語）を自動で検出し、その言語で対話を開始します。
-   **探求的な質問**: AIを活用し、対話の文脈に応じてユーザーの思考を促す質問を生成します。
-   **セッションの振り返り**: `!reflect` コマンドで、いつでも対話の途中経過や発見を要約して確認できます。
-   **対話の記録**: `!finish` コマンドでセッションを終了すると、対話の全記録がMarkdownファイルとして保存されます。

さらに、会話内容から得られた知識や洞察を、RAG（Retrieval-Augmented Generation）システムで活用しやすいように構造化されたJSONL形式(`.jsonl`)のファイルとしても同時に出力します。

## 動作環境

-   Python 3.8 以上

## セットアップ方法

1.  **リポジトリのクローンまたはダウンロード**
    
    このプロジェクトのファイルをローカル環境に配置します。
    
2.  **必要なライブラリのインストール**
    
    ターミナルで以下のコマンドを実行し、必要なPythonライブラリをインストールします。
    
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **環境変数の設定**
    
    プロジェクトのルートディレクトリに `.env` という名前のファイルを作成し、以下の内容を記述します。
    
    ```
    DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```
    
    -   `YOUR_DISCORD_BOT_TOKEN`: あなたのDiscord Botのトークンに置き換えてください。トークンは [Discord Developer Portal](https://discord.com/developers/applications) で取得できます。
    -   `YOUR_OPENAI_API_KEY`: あなたのOpenAI APIキーに置き換えてください。キーは [OpenAI Platform](https://platform.openai.com/api-keys) で取得できます。
    

## Botの起動方法

セットアップが完了したら、ターミナルで以下のコマンドを実行してBotを起動します。

```bash
python nani_bot.py
```

コンソールに `🧠 EL has started!` と表示されれば成功です。

## コマンド一覧

### `!explore [トピック]`

新しい思考の探求セッションを開始します。Botはトピックの言語を自動で判別し、その言語であなた専用のスレッドを作成して最初の質問を投げかけます。

**使用例:**

-   **日本語での開始:**
    
    ```
    !explore 最近感動したこと
    ```
    
-   **英語での開始:**
    
    ```
    !explore something that moved me recently
    ```
    

### `!reflect`

現在進行中のセッションの内容を振り返ります。これまでの対話から得られた「主な発見」や「深まった理解」などを、セッションで使われている言語でまとめたメッセージが送信されます。

### `!finish`

セッションを終了し、対話の全記録をMarkdownファイル(`.md`)として出力します。この記録もセッションの言語で生成されます。

さらに、会話内容から得られた知識や洞察を、RAG（Retrieval-Augmented Generation）システムで活用しやすいように構造化されたJSONL形式(`.jsonl`)のファイルとしても同時に出力します。

## ファイル構成（主要）

- `agent/` : 暗黙知抽出エージェント (FastAPI) の実装
- `tests/` : pytest ユニットテスト
- `docs/OPERATIONS.md` : 運用 Runbook
- `README.md` : 本ドキュメント
- `requirements.txt` : ランタイム依存ライブラリ
- `dev-requirements.txt` : 開発 / CI 用ライブラリ
- `Makefile` : ローカル CI コマンド (`make ci` など) 