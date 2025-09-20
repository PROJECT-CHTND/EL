# EL エージェント実装計画 – **v2 仮説検証型アーキテクチャ**

> 2024-XX 更新 – 反復的推論ループ・動的知識統合・戦略的質問生成・ドキュメント自動生成を中核に据えた新バージョン

---
## 0. 変更概要

| v1 (旧計画) | v2 (新計画) |
|-------------|-------------|
| 直線的 7 ステージパイプライン | **Reasoning Loop** を中心にした反復的推論アーキテクチャ |
| KG 抽出→スロット→質問テンプレ | 仮説生成→知識統合→証拠抽出→仮説更新→戦略的質問 |
| 信頼度 = logprob × weight | **多次元信頼度** (言語・文脈・事実性・ソース信頼) |
| 質問 = スロット充填 | **戦略別質問** (Surface / Probing / Contradict / Hypothetical / Comparative) |
| 出力 = 次の質問のみ | **追加出力**: 構造化ドキュメント (手順書・FAQ 等) |

---
## 1. 全体アーキテクチャ

```
User ↔ Adaptive Dialogue Manager ↔ Reasoning Loop (max N iterations)
                                    ├─ Hypothesis Generation (LLM)
                                    ├─ Dynamic Knowledge Integrator (Web/VDB/KG)
                                    ├─ Evidence Extraction (LLM Function)
                                    └─ Confidence Evaluator (python)
                                   ↓
                         Question Strategist (LLM)
                                   ↓
                             Userへ次の質問
                                   ↓
                         Document Synthesizer (LLM) → Markdown / JSON
```

### 主要コンポーネント

| 番号 | コンポーネント | 技術 | 備考 |
|------|----------------|------|------|
| 1 | **Adaptive Dialogue Manager** | GPT-4o / python | ユーザー応答を感情・情報量で分類し、ループパラメータを調整 |
| 2 | **Reasoning Loop** | asyncio | 反復制御オーケストレータ (最大 3〜5 周) |
| 2-a | Hypothesis Generation | GPT-4o (high-T) | 会話ログ + KG から仮説を生成 |
| 2-b | Dynamic Knowledge Integrator | Web Search API, VectorDB(Qdrant), Neo4j | 仮説に基づく情報収集 |
| 2-c | Evidence Extraction | GPT-4o Function-Calling (`save_kv`) | 証拠を KG 形式へ抽出 |
| 2-d | Confidence Evaluator | python | 言語/文脈/事実/矛盾/ソース の 5 軸でスコアリング |
| 3 | **Question Strategist** | GPT-4o | 会話深度に応じて戦略を切替え質問生成 |
| 4 | **Document Synthesizer** | GPT-4o | 完成 KG + 仮説から手順書/FAQ 等を Markdown で生成 |
| 5 | **Stores** | Redis, Neo4j, VectorDB | セッション状態 / 永続 KG / ドキュメントコーパス |

---
## 2. ディレクトリ構成 (変更点のみ)

```
agent/
  reasoning/            # new: Reasoning Loop & sub-modules
    __init__.py
    loop.py             # Iterative controller
    hypotheses.py       # Hypothesis data model
    knowledge.py        # Integrator
    confidence.py       # Multi-dim evaluator
  docsynth/             # new: Structured Document Synthesizer
    __init__.py
    synth.py
  prompts/
    reasoning/          # system prompts per sub-module
    docs/
```

---
## 3. マイルストーン

| 週 | Deliverable | 主要タスク |
|---|---|---|
| 1 | **MVP Reasoning Loop** | loop.py + HypothesisGeneration Prompt + Redis session |
| 2 | **Dynamic Knowledge Integrator** | Web & VDB 接続、関連度スコアリング |
| 3 | **Evidence Extraction v2** | Function-calling + Neo4j ingest、Confidence Evaluator v1 |
| 4 | **Question Strategist** | 戦略 YAML + Prompt 実装、Discord Bot 統合 |
| 5 | **Document Synthesizer** | 手順書テンプレート prompt、Markdown 出力 |
| 6 | **E2E テスト & Metrics** | pytest + Prometheus、Docker Compose update |

---
## 4. API / CLI エンドポイント

| 名称 | パス | 概要 |
|------|------|------|
| `POST /dialogue` | (FastAPI) | ユーザー発話を送信 → 次質問 + interim KG 返却 |
| `POST /document` | | 対話終了後、指定フォーマットのドキュメントを生成 |
| `scripts/pipeline_cli_v2.py` | CLI | テキストファイル → Reasoning Loop → 質問 & ドキュメントを標準出力 |

---
## 5. 信頼度スキーマ (例)

```json
{
  "entity_id": "123",
  "confidence": {
    "linguistic": 0.85,
    "contextual": 0.78,
    "factual": 0.92,
    "consistency": 0.88,
    "source": 0.60,
    "overall": 0.81
  }
}
```

---
## 6. テスト戦略更新

1. **Unit**: 各 Reasoning sub-module の JSON スキーマ妥当性
2. **Loop Simulation**: モック LLM で 3 ループ完走を検証
3. **Doc Synth**: 入力 KG → Markdown のスナップショットテスト

---
## 7. リスク & 緩和策 (追加)
| リスク | 対応 |
|-------|------|
| Reasoning Loop が収束しない | 最大反復数とタイムアウトを設定、fallback 質問生成へ切替 |
| Web Search API コスト | キャッシュ & レート制限、社内 KB 優先検索 |
| ドキュメント品質バラツキ | QA チェック Prompt + 人手レビュー ワークフロー |

---
## 8. まとめ

新バージョンでは「仮説→検証→質問→ドキュメント化」の反復プロセスを中心に据え、EL を **有能なファシリテータ兼ナレッジワーカー** として位置付けます。既存コンポーネントはサブモジュールとして再利用しつつ、より深い思考と再利用可能な成果物を同時に実現します。
