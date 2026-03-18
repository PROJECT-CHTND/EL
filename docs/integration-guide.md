# EL 統合ガイド

## 概要

ELはREST APIを通じてあらゆるシステムと統合できます。このガイドでは、代表的な統合パターンを紹介します。

---

## 1. 社内チャットボットとの統合

既存のチャットボット（Slack Bot、Teams Bot等）にELの質問機能を組み込むことで、自然な対話を通じた知識収集が可能になります。

### Slack Bot 統合例

```python
from slack_bolt import App
from el_sdk import ELClient

app = App(token="xoxb-your-token")
el = ELClient(api_key="your-license-key")

# アクティブなセッションを保持
active_sessions = {}

@app.message("引き継ぎ開始")
def start_handover(message, say):
    user_id = message["user"]
    session = el.create_session(topic=f"{user_id}の引き継ぎ")
    active_sessions[user_id] = session

    question = session.get_next_question()
    say(f"引き継ぎセッションを開始します。\n\n🔍 {question.text}")

@app.message("")
def handle_response(message, say):
    user_id = message["user"]
    session = active_sessions.get(user_id)
    if not session:
        return

    # 現在の質問に回答
    # （質問IDの管理は簡略化しています）
    session.respond(current_question_id, message["text"])

    # 次の質問を取得
    next_q = session.get_next_question()
    if next_q:
        say(f"ありがとうございます。次の質問です。\n\n🔍 {next_q.text}")
    else:
        summary = session.get_summary()
        say(f"✅ インタビュー完了！{summary.total_facts}件の知識を蓄積しました。")
```

---

## 2. 既存ナレッジベースとの連携

Confluence、Notion、SharePoint等の既存ドキュメントをELに取り込み、不足情報を特定できます。

### 連携フロー

```
既存ナレッジベース
    ↓ エクスポート（PDF / テキスト）
EL Document Pipeline
    ↓ 自動解析・ファクト抽出
矛盾・不足の検知
    ↓
補完が必要な情報について質問生成
    ↓
担当者へインタビュー
    ↓
補完された知識をナレッジベースに反映
```

---

## 3. Webhook連携

ELのイベントをトリガーにして外部システムと連携できます。

### 活用例

- **矛盾検知時にSlack通知**: `inconsistency.detected` イベントでアラート送信
- **ドキュメント処理完了時に次のワークフローを起動**: `document.processed` イベント
- **セッション完了時にレポート自動生成**: `session.completed` イベント

---

## 4. フロントエンド統合

EL APIを使って独自のUIを構築できます。`el_frontend/` にReactベースの参考実装があります。

### 主要なUI要素

- **チャットインターフェース**: 質問と回答の対話UI
- **ナレッジマップビューア**: `/knowledge-map` APIデータの可視化
- **ファクトブラウザ**: 蓄積された知識の閲覧・検索
- **サマリーダッシュボード**: カバレッジスコアと未解決ギャップの表示
