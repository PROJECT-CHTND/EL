<div align="center">

# 🔍 EL — Eager Learner

**質問する力で、組織の知識を引き出すAIエージェント**

*Question-Driven Knowledge Extraction Agent*

[![License](https://img.shields.io/badge/License-Proprietary-blue?style=for-the-badge)](#license)
[![API](https://img.shields.io/badge/API-v1.0-green?style=for-the-badge)](#api-reference)

<br />

> **「答えるAI」ではなく、「聞くAI」。**
> ELは的確な質問を通じて、人の中にある暗黙知を引き出し、組織の知的資産として構造化します。

<br />

</div>

---

## 🎯 ELとは

**Eager Learner（EL）** は、ナレッジマネジメントに特化した質問駆動型AIエージェントです。

従来のAIは「答える」ことに注力していますが、ELは「聞く」ことに特化しています。ドキュメントや会話から情報の矛盾・不足を自動検知し、的確な質問を生成することで、ベテラン社員の暗黙知を効率的に引き出し、構造化された知識資産として蓄積します。

### 解決する課題

- **業務引き継ぎの属人化** — ベテラン社員の退職により消失する暗黙知を、AIが体系的にインタビューして形式知化
- **ドキュメントの矛盾・欠落** — 既存資料の不整合を自動検知し、補完すべき情報を特定
- **知識のサイロ化** — 部署や案件ごとに散在する知識を統合的に構造化

---

## ✨ 主要機能

### 🧠 インテリジェント質問生成
蓄積された知識と新規情報を照合し、矛盾・不足・曖昧さを自動検知。文脈に応じた深掘り質問を生成します。知識が蓄積されるほど、より専門的で的確な質問が可能になります。

### 📊 Fact Ledger（知識台帳）
抽出された情報を「誰が・何を・いつ・どうした」の構造化データとして自動整理。事実同士の関連性や時系列を保持し、矛盾検知の基盤となります。

### 🗺️ ナレッジマップ
蓄積された知識を視覚的にマッピング。情報の重要度を光の強さで、知識間の関連性を接続線で表現し、組織の知的資産の全体像を俯瞰できます。

### 📄 ドキュメント取り込み
PDF・テキストファイルをアップロードするだけで、自動的にファクト抽出・分類・タグ付けを実行。既存ドキュメントからの知識移行をスムーズに行えます。

---

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│                    Your Application                  │
│              (Web UI / Chat Bot / etc.)               │
└──────────────────────┬──────────────────────────────┘
                       │  REST API
                       ▼
┌─────────────────────────────────────────────────────┐
│                    EL API Gateway                    │
│              OpenAPI / Swagger 準拠                   │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌─────────┐ ┌──────────┐
   │ Interview  │ │  Fact   │ │ Document │
   │  Engine    │ │ Ledger  │ │ Pipeline │
   │            │ │         │ │          │
   │ 質問生成   │ │ 知識管理 │ │ 文書解析  │
   │ 矛盾検知   │ │ 構造化  │ │ 自動分類  │
   │ 深掘り制御  │ │ 検索    │ │ タグ付け  │
   └────────────┘ └─────────┘ └──────────┘
          │            │            │
          └────────────┼────────────┘
                       ▼
              ┌─────────────────┐
              │  EL Core Engine │
              │   (Licensed)    │
              └─────────────────┘
```

**EL Core Engine** はライセンス提供のコア技術です。APIを通じてすべての機能にアクセスでき、お客様のアプリケーションやUIと自由に統合できます。

---

## 🚀 クイックスタート

### 前提条件

- Docker & Docker Compose
- ELライセンスキー（[お問い合わせ](#contact)から取得）

### 起動

```bash
# リポジトリをクローン
git clone https://github.com/your-org/el-knowledge.git
cd el-knowledge

# 環境変数を設定
cp .env.example .env
# .env にライセンスキーとLLM APIキーを設定

# 起動
docker compose up -d
```

起動後、`http://localhost:8080` でデモUIにアクセスできます。

### 最初のインタビューセッション

```python
from el_sdk import ELClient

client = ELClient(base_url="http://localhost:8000", api_key="your-license-key")

# セッション開始
session = client.create_session(topic="プロジェクトAの引き継ぎ")

# ドキュメントをアップロード
session.upload_document("handover_notes.pdf")

# ELが質問を生成
question = session.get_next_question()
print(question.text)
# => "引き継ぎ資料では3月にシステム移行が完了したとありますが、
#     移行後の検証フェーズについての記載がありません。
#     検証はどのように実施されましたか？"

# 回答を送信
session.respond(question.id, "検証は4月に2週間かけて実施し...")

# 蓄積された知識を確認
facts = session.get_facts()
```

---

## 📖 API Reference

詳細なAPI仕様は [docs/api-reference.md](docs/api-reference.md) を参照してください。

### 主要エンドポイント

| Method | Endpoint | 説明 |
|--------|----------|------|
| `POST` | `/sessions` | インタビューセッションを作成 |
| `POST` | `/sessions/{id}/documents` | ドキュメントをアップロード |
| `GET` | `/sessions/{id}/questions/next` | 次の質問を取得 |
| `POST` | `/sessions/{id}/responses` | 質問への回答を送信 |
| `GET` | `/sessions/{id}/facts` | 蓄積されたファクトを取得 |
| `GET` | `/sessions/{id}/summary` | ナレッジサマリーを取得 |
| `GET` | `/sessions/{id}/knowledge-map` | ナレッジマップデータを取得 |
| `POST` | `/webhooks` | Webhook通知を登録 |

---

## 📁 リポジトリ構成

```
el-knowledge/
├── README.md                 # このファイル
├── docker-compose.yml        # 起動設定（EL Core Engineはイメージとして取得）
├── .env.example              # 環境変数テンプレート
├── CHANGELOG.md              # 更新履歴
├── docs/
│   ├── architecture.md       # アーキテクチャ詳細
│   ├── api-reference.md      # API仕様書
│   ├── deployment-guide.md   # デプロイガイド
│   └── integration-guide.md  # 統合ガイド
├── sdk/
│   └── python/               # Python SDK
│       ├── el_sdk/
│       └── setup.py
├── examples/
│   ├── basic_interview.py    # 基本的なインタビュー
│   ├── document_upload.py    # ドキュメント取り込み
│   └── chatbot_integration/  # チャットボット統合例
└── el_frontend/              # デモUI（参考実装）
    ├── src/
    └── package.json
```

---

## 🔄 更新履歴

最新の更新内容は [CHANGELOG.md](CHANGELOG.md) を参照してください。

ELライセンスには、LLMモデル世代交代への追従、RAGパイプラインの継続的最適化、プロンプトエンジニアリングのアップデートが含まれます。

---

## 📋 動作要件

| 項目 | 要件 |
|------|------|
| Docker | 20.10+ |
| メモリ | 4GB以上（推奨8GB） |
| ストレージ | 10GB以上 |
| LLM API | OpenAI API または Azure OpenAI |
| ネットワーク | LLM APIへのアウトバウンド接続 |

オンプレミス環境での利用やローカルLLMとの連携については[お問い合わせ](#contact)ください。

---

## <a name="contact"></a>📬 お問い合わせ

ELの導入・ライセンスに関するお問い合わせ:

- **Email**: contact@example.com
- **Web**: https://example.com/el

技術的なご質問やPoCのご相談も承っています。

---

## <a name="license"></a>📄 ライセンス

EL Core Engineはプロプライエタリソフトウェアです。利用にはライセンス契約が必要です。

デモUI（`el_frontend/`）およびSDK（`sdk/`）はMIT Licenseで提供されています。

詳細は [LICENSE](LICENSE) を参照してください。

---

<div align="center">

**Built with curiosity. Powered by questions.**

*EL — 知識を「聞き出す」AI*

</div>
