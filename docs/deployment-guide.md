# EL デプロイガイド

## デプロイメントモデル

ELは3つのデプロイ形態に対応しています。環境のセキュリティ要件に応じて選択してください。

### 1. SaaS（マネージドサービス）

ELが管理するクラウド環境を利用します。インフラの管理は不要で、最も迅速に導入できます。

- **利点**: 即時利用開始、インフラ管理不要、自動アップデート
- **要件**: インターネット接続
- **推奨**: 小〜中規模組織、迅速な導入

### 2. 顧客VPCデプロイ

お客様のAWS/Azure/GCP環境にELをデプロイします。データは顧客環境内に保持されます。

- **利点**: データの完全な管理、既存インフラとの統合
- **要件**: Docker対応環境、4GB以上のメモリ
- **推奨**: データを社外に出せない企業

### 3. オンプレミスデプロイ

お客様の物理サーバーにELをデプロイします。

- **利点**: 完全な物理的データ管理
- **要件**: Docker対応Linux環境、ネットワーク要件を参照
- **推奨**: 金融機関、官公庁等

---

## Docker Compose によるデプロイ

### 前提条件

- Docker 20.10+
- Docker Compose 2.0+
- ELライセンスキー
- LLM API キー（OpenAI または Azure OpenAI）

### 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/your-org/el-knowledge.git
cd el-knowledge

# 2. 環境変数を設定
cp .env.example .env
# .env を編集してライセンスキーとAPIキーを設定

# 3. EL Core Engineイメージを取得（認証が必要）
docker login ghcr.io
docker pull ghcr.io/your-org/el-core:latest

# 4. 起動
docker compose up -d

# 5. 稼働確認
curl http://localhost:8000/health
```

### ネットワーク要件

| 方向 | 宛先 | ポート | 用途 |
|------|------|--------|------|
| Outbound | api.openai.com | 443 | LLM API通信 |
| Outbound | ghcr.io | 443 | イメージ取得（初回のみ） |
| Outbound | license.example.com | 443 | ライセンス検証 |
| Inbound | - | 8000 | EL API |
| Inbound | - | 8080 | デモUI（オプション） |

### データの永続化

PostgreSQLのデータはDockerボリューム `pgdata` に保存されます。
バックアップは以下のコマンドで取得できます。

```bash
docker compose exec postgres pg_dump -U el el_knowledge > backup.sql
```

---

## 本番環境での推奨設定

### セキュリティ

- TLS/SSL終端を設置（リバースプロキシ推奨）
- データベースパスワードの強化
- ファイアウォールルールの適用
- 定期的なバックアップの設定

### モニタリング

EL APIは `/health` エンドポイントでヘルスチェックを提供します。
Prometheus互換のメトリクスは `/metrics` で取得できます。

### アップデート

```bash
# 最新イメージを取得
docker compose pull

# サービスを再起動（データは保持されます）
docker compose up -d
```

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `invalid_license` エラー | ライセンスキーが無効 | .envのEL_LICENSE_KEY を確認 |
| `license_expired` エラー | ライセンス期限切れ | ライセンスの更新をお問い合わせください |
| DB接続エラー | PostgreSQLが未起動 | `docker compose up postgres` を先に実行 |
| LLM APIエラー | APIキーが無効またはクォータ超過 | LLM_API_KEY を確認 |

詳細なサポートは contact@example.com までお問い合わせください。
