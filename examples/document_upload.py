"""
EL ドキュメント取り込み例

既存ドキュメントをELにアップロードし、自動的にファクト抽出・矛盾検知を行う例です。
処理完了後、検知された矛盾と不足情報を確認できます。
"""

import time

from el_sdk import ELClient


def main():
    client = ELClient(
        base_url="http://localhost:8000",
        api_key="your-license-key",
    )

    # セッションを作成
    session = client.create_session(
        topic="既存ドキュメントの分析",
        description="過去の議事録・引き継ぎ資料を取り込み、情報の整合性を検証",
    )
    print(f"セッション作成: {session.id}\n")

    # 複数ドキュメントをアップロード
    documents = [
        ("meeting_notes_may.pdf", "5月度 研究会議事録"),
        ("meeting_notes_june.pdf", "6月度 研究会議事録"),
        ("handover_doc.pdf", "引き継ぎ資料"),
    ]

    doc_ids = []
    for filepath, description in documents:
        try:
            result = session.upload_document(filepath, description=description)
            doc_ids.append(result["id"])
            print(f"📄 アップロード: {filepath} → {result['id']}")
        except FileNotFoundError:
            print(f"⚠️  ファイルが見つかりません: {filepath}")

    if not doc_ids:
        print("アップロードされたドキュメントがありません。")
        return

    # 処理完了を待機
    print("\n⏳ ドキュメントを処理中...")
    all_done = False
    while not all_done:
        time.sleep(5)
        all_done = True
        for doc_id in doc_ids:
            status = client._request("GET", f"/sessions/{session.id}/documents/{doc_id}")
            if status["status"] != "completed":
                all_done = False
                print(f"   {doc_id}: {status['status']}...")

    print("✅ すべてのドキュメントの処理が完了しました\n")

    # 抽出結果を確認
    facts = session.get_facts()
    print(f"📊 抽出されたファクト: {len(facts)}件")

    # カテゴリ別に集計
    categories = {}
    for fact in facts:
        cat = fact.category
        categories[cat] = categories.get(cat, 0) + 1

    print("\nカテゴリ別内訳:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}件")

    # サマリーを確認
    summary = session.get_summary()

    if summary.inconsistencies:
        print(f"\n❗ 検知された矛盾: {len(summary.inconsistencies)}件")
        for inc in summary.inconsistencies:
            print(f"  • {inc['description']}")

    if summary.remaining_gaps:
        print(f"\n⚠️  情報の不足: {len(summary.remaining_gaps)}件")
        for gap in summary.remaining_gaps:
            print(f"  • {gap}")

    # 矛盾・不足に基づいてELが生成した質問を確認
    print(f"\n🔍 ELが生成した質問:")
    for i in range(5):
        question = session.get_next_question()
        if question is None:
            break
        print(f"  [{question.type}] {question.text}")
        session.skip_question(question.id)

    print(f"\n💡 これらの質問に回答するには basic_interview.py を使用してください。")
    print(f"   セッションID: {session.id}")


if __name__ == "__main__":
    main()
