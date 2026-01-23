#!/usr/bin/env python3
"""Seed demo data for EL demonstration.

This script populates the knowledge graph with sample insights
to demonstrate the "remembering" and "contradiction detection" features.

Usage:
    uv run python scripts/seed_demo_data.py
    uv run python scripts/seed_demo_data.py --clear  # Clear existing data first
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add packages to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core", "src"))

from dotenv import load_dotenv

load_dotenv()

from el_core.schemas import Domain, Insight
from el_core.stores.kg_store import KnowledgeGraphStore


# Demo insights for daily work scenario
DAILY_WORK_INSIGHTS = [
    # 1週間前のセッション: PRレビューの課題
    Insight(
        subject="ユーザー",
        predicate="reported_issue",
        object="PRレビューのフィードバックに時間がかかり、レビュー待ちが溜まっている",
        confidence=0.9,
        domain=Domain.DAILY_WORK,
        timestamp=datetime.now() - timedelta(days=7),
    ),
    Insight(
        subject="ユーザー",
        predicate="committed_to",
        object="PRレビューを24時間以内に完了することを目標にする",
        confidence=0.85,
        domain=Domain.DAILY_WORK,
        timestamp=datetime.now() - timedelta(days=7),
    ),
    Insight(
        subject="ユーザー",
        predicate="prefers",
        object="コードレビューは午前中に集中して行う方が効率が良い",
        confidence=0.8,
        domain=Domain.DAILY_WORK,
        timestamp=datetime.now() - timedelta(days=5),
    ),
    # 3日前のセッション: プロジェクト進捗
    Insight(
        subject="新機能プロジェクト",
        predicate="has_status",
        object="デザインレビューが完了し、実装フェーズに入った",
        confidence=0.9,
        domain=Domain.DAILY_WORK,
        timestamp=datetime.now() - timedelta(days=3),
    ),
    Insight(
        subject="ユーザー",
        predicate="concerned_about",
        object="テスト工数が不足する可能性がある",
        confidence=0.75,
        domain=Domain.DAILY_WORK,
        timestamp=datetime.now() - timedelta(days=3),
    ),
]

# Demo insights for postmortem scenario
POSTMORTEM_INSIGHTS = [
    Insight(
        subject="先月の障害",
        predicate="root_cause",
        object="設定ファイルの変更が本番環境に反映されていなかった",
        confidence=0.95,
        domain=Domain.POSTMORTEM,
        timestamp=datetime.now() - timedelta(days=14),
    ),
    Insight(
        subject="ユーザー",
        predicate="implemented",
        object="設定変更のレビュープロセスを導入した",
        confidence=0.9,
        domain=Domain.POSTMORTEM,
        timestamp=datetime.now() - timedelta(days=14),
    ),
    Insight(
        subject="インシデント対応",
        predicate="learned_that",
        object="アラートの閾値が厳しすぎて誤検知が多かった",
        confidence=0.85,
        domain=Domain.POSTMORTEM,
        timestamp=datetime.now() - timedelta(days=10),
    ),
]

# Demo insights for recipe scenario  
RECIPE_INSIGHTS = [
    Insight(
        subject="カレー",
        predicate="tip",
        object="玉ねぎを飴色になるまで炒めると甘みが出る",
        confidence=0.9,
        domain=Domain.RECIPE,
        timestamp=datetime.now() - timedelta(days=20),
    ),
    Insight(
        subject="ユーザー",
        predicate="prefers",
        object="辛さは中辛程度が好み",
        confidence=0.85,
        domain=Domain.RECIPE,
        timestamp=datetime.now() - timedelta(days=20),
    ),
]

ALL_DEMO_INSIGHTS = DAILY_WORK_INSIGHTS + POSTMORTEM_INSIGHTS + RECIPE_INSIGHTS


async def clear_all_data(kg_store: KnowledgeGraphStore) -> None:
    """Clear all existing data from the knowledge graph."""
    print("Clearing existing data...")
    async with kg_store.driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    print("✓ All data cleared")


async def seed_demo_data(kg_store: KnowledgeGraphStore) -> None:
    """Seed the knowledge graph with demo insights."""
    print(f"\nSeeding {len(ALL_DEMO_INSIGHTS)} demo insights...")
    
    for i, insight in enumerate(ALL_DEMO_INSIGHTS, 1):
        try:
            insight_id = await kg_store.save_insight(insight)
            domain_emoji = {
                Domain.DAILY_WORK: "💼",
                Domain.POSTMORTEM: "🔍",
                Domain.RECIPE: "🍳",
            }.get(insight.domain, "💭")
            
            print(f"  {domain_emoji} [{i}/{len(ALL_DEMO_INSIGHTS)}] {insight.subject}: {insight.object[:50]}...")
        except Exception as e:
            print(f"  ⚠ Failed to save insight {i}: {e}")
    
    print(f"\n✓ Demo data seeded successfully!")


async def verify_data(kg_store: KnowledgeGraphStore) -> None:
    """Verify the seeded data by running test queries."""
    print("\n--- Verification ---")
    
    # Test search for daily work
    results = await kg_store.search("PRレビュー", limit=5)
    print(f"\n🔍 Search 'PRレビュー': {len(results)} results")
    for r in results:
        print(f"   - {r.subject}: {r.object[:40]}...")
    
    # Test search for postmortem
    results = await kg_store.search("障害", limit=5)
    print(f"\n🔍 Search '障害': {len(results)} results")
    for r in results:
        print(f"   - {r.subject}: {r.object[:40]}...")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed demo data for EL demonstration")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before seeding")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    args = parser.parse_args()
    
    # Check for Neo4j configuration
    neo4j_uri = os.getenv("NEO4J_URI")
    if not neo4j_uri:
        print("❌ NEO4J_URI environment variable not set")
        print("   Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in .env")
        sys.exit(1)
    
    print(f"Connecting to Neo4j at {neo4j_uri}...")
    
    kg_store = KnowledgeGraphStore()
    
    try:
        await kg_store.connect()
        print("✓ Connected to Neo4j")
        
        if args.verify_only:
            await verify_data(kg_store)
        else:
            if args.clear:
                await clear_all_data(kg_store)
            
            await kg_store.setup_indexes()
            await seed_demo_data(kg_store)
            await verify_data(kg_store)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await kg_store.close()
    
    print("\n" + "="*50)
    print("Demo data is ready!")
    print("="*50)
    print("""
デモシナリオ例:

[日報シナリオ]
1. トピック: 「今日の進捗を整理したい」
2. 「PRレビューを何件かやりました」と入力
   → ELが過去の「PRレビューに時間がかかる課題」を参照
   → 「以前、レビューに時間がかかるとおっしゃっていましたが、
      今回はどうでしたか？」と質問

[障害振り返りシナリオ]
1. トピック: 「今日起きた障害について整理したい」
2. 「設定ミスで障害が起きた」と入力
   → ELが過去の「設定変更レビュープロセス導入」を参照
   → 「先月、設定変更のレビュープロセスを導入されましたが、
      今回はそのプロセスは機能しましたか？」と質問
""")


if __name__ == "__main__":
    asyncio.run(main())
