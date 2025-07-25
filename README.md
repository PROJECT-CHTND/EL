# Nani - A Bilingual Thinking Partner

## 概要

Naniは、Discord上で対話を通じてユーザーの考えや洞察を深める手助けをするためのBotです。指定されたトピックについて、Botが知識に基づいた探求的な質問を投げかけ、ユーザーが自分の考えを整理し、新しい視点を発見することをサポートします。

**日本語と英語の両方に対応しています。**

## 主な機能

-   **バイリンガル対話**: `!explore` コマンドで指定したトピックの言語（日本語または英語）を自動で検出し、その言語で対話を開始します。
-   **探求的な質問**: AIを活用し、対話の文脈に応じてユーザーの思考を促す質問を生成します。
-   **セッションの振り返り**: `!reflect` コマンドで、いつでも対話の途中経過や発見を要約して確認できます。
-   **対話の記録**: `!finish` コマンドでセッションを終了すると、対話の全記録がMarkdownファイルとして保存されます。

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

コンソールに `🧠 Nani has started!` と表示されれば成功です。

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

## ファイル構成

-   `nani_bot.py`: Botのメインプログラムです。
-   `prompts.py`: AIに与える指示（プロンプト）を管理するファイルです。対話の進行やRAGデータ生成のためのテンプレートが含まれます。
-   `requirements.txt`: 必要なPythonライブラリのリストです。
-   `.env` (作成が必要): APIキーなどの機密情報を保存するファイルです。
-   `README.md`: このファイルです。Botの概要や使い方を説明しています。 