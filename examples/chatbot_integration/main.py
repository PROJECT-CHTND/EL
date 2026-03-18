"""
EL チャットボット統合例

FastAPIベースのWebhookレシーバーと、Slack Botとの統合サンプルです。
ELのイベントをリアルタイムで受信し、Slackに通知します。
"""

from fastapi import FastAPI, Request
from el_sdk import ELClient

app = FastAPI(title="EL Chatbot Integration")

el = ELClient(
    base_url="http://localhost:8000",
    api_key="your-license-key",
)

# ============================================
# Webhook受信エンドポイント
# ============================================

@app.post("/webhooks/el")
async def receive_el_webhook(request: Request):
    """ELからのWebhookイベントを受信"""
    payload = await request.json()
    event_type = payload.get("event")
    data = payload.get("data", {})

    if event_type == "inconsistency.detected":
        await handle_inconsistency(data)
    elif event_type == "document.processed":
        await handle_document_processed(data)
    elif event_type == "session.completed":
        await handle_session_completed(data)

    return {"status": "ok"}


async def handle_inconsistency(data: dict):
    """矛盾検知時の処理"""
    print(f"❗ 矛盾を検知: {data['description']}")
    # Slack通知やメール送信など
    # await send_slack_message(
    #     channel="#knowledge-alerts",
    #     text=f"⚠️ ELが矛盾を検知しました: {data['description']}"
    # )


async def handle_document_processed(data: dict):
    """ドキュメント処理完了時の処理"""
    print(f"📄 処理完了: {data['filename']} → {data['facts_extracted']}件のファクト抽出")


async def handle_session_completed(data: dict):
    """セッション完了時の処理"""
    print(f"✅ セッション完了: {data['session_id']}")
    print(f"   ファクト数: {data['total_facts']}")
    print(f"   カバレッジ: {data['coverage_score']:.0%}")


# ============================================
# チャットボット用APIエンドポイント
# ============================================

@app.post("/chat/start")
async def start_session(request: Request):
    """外部チャットボットからセッションを開始"""
    body = await request.json()
    session = el.create_session(
        topic=body.get("topic", "新規セッション"),
        description=body.get("description", ""),
    )
    question = session.get_next_question()
    return {
        "session_id": session.id,
        "first_question": question.text if question else None,
    }


@app.post("/chat/respond")
async def respond_to_question(request: Request):
    """ユーザーの回答を処理し、次の質問を返す"""
    body = await request.json()
    session = el.get_session(body["session_id"])

    result = session.respond(
        question_id=body["question_id"],
        response_text=body["response_text"],
    )

    next_question = session.get_next_question()

    return {
        "facts_extracted": result["facts_extracted"],
        "next_question": {
            "id": next_question.id,
            "text": next_question.text,
            "type": next_question.type,
        } if next_question else None,
        "session_complete": next_question is None,
    }


@app.post("/chat/summary")
async def get_summary(request: Request):
    """セッションのサマリーを取得"""
    body = await request.json()
    session = el.get_session(body["session_id"])
    summary = session.get_summary()

    return {
        "total_facts": summary.total_facts,
        "coverage_score": summary.coverage_score,
        "key_findings": summary.key_findings,
        "remaining_gaps": summary.remaining_gaps,
    }


# ============================================
# 起動
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
