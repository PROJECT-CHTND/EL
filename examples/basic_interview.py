"""
EL 基本的なインタビュー例

このスクリプトは、ELを使った基本的なインタビューセッションの流れを示します。
ドキュメントをアップロードし、ELが生成する質問に対話的に回答します。
"""

from el_sdk import ELClient


def main():
    # クライアントを初期化
    client = ELClient(
        base_url="http://localhost:8000",
        api_key="your-license-key",
    )

    # ヘルスチェック
    health = client.health_check()
    print(f"EL API Status: {health['status']}")

    # セッションを作成
    session = client.create_session(
        topic="プロジェクトAの引き継ぎ",
        description="2024年度のシステム移行プロジェクトに関する知識の引き継ぎ",
        tags=["プロジェクトA", "システム移行"],
    )
    print(f"\nセッション作成: {session.id}")

    # ドキュメントをアップロード（オプション）
    # session.upload_document("path/to/handover_notes.pdf")
    # print("ドキュメントを処理中...")

    # インタビューループ
    print("\n" + "=" * 60)
    print("EL インタビューセッション開始")
    print("'quit' で終了、'skip' で質問をスキップ")
    print("=" * 60)

    while True:
        # 次の質問を取得
        question = session.get_next_question()

        if question is None:
            print("\n✅ すべての質問が完了しました。")
            break

        # 質問を表示
        print(f"\n🔍 [{question.type}] {question.text}")

        if question.context.get("detected_gap"):
            print(f"   💡 背景: {question.context['detected_gap']}")

        # ユーザーの回答を受け取る
        answer = input("\n📝 回答: ").strip()

        if answer.lower() == "quit":
            break

        if answer.lower() == "skip":
            session.skip_question(question.id)
            print("   ⏭️ スキップしました")
            continue

        # 回答を送信
        result = session.respond(question.id, answer)
        print(f"   ✅ {result['facts_extracted']}件のファクトを抽出しました")

    # サマリーを表示
    print("\n" + "=" * 60)
    print("📊 セッションサマリー")
    print("=" * 60)

    summary = session.get_summary()
    print(f"\n蓄積ファクト数: {summary.total_facts}")
    print(f"カバレッジスコア: {summary.coverage_score:.0%}")

    if summary.key_findings:
        print("\n📌 主要な発見:")
        for finding in summary.key_findings:
            print(f"  • {finding}")

    if summary.remaining_gaps:
        print("\n⚠️ 未解決のギャップ:")
        for gap in summary.remaining_gaps:
            print(f"  • {gap}")

    if summary.inconsistencies:
        print("\n❗ 検知された矛盾:")
        for inc in summary.inconsistencies:
            print(f"  • {inc['description']} [{inc['status']}]")


if __name__ == "__main__":
    main()
