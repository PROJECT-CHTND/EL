# EL API Reference

**Base URL**: `http://localhost:8000/api/v1`

**認証**: すべてのリクエストに `Authorization: Bearer {license_key}` ヘッダーが必要です。

---

## Sessions（セッション管理）

### セッションを作成

```
POST /sessions
```

新しいインタビューセッションを開始します。

**リクエスト**

```json
{
  "topic": "プロジェクトAの引き継ぎ",
  "description": "2024年度のシステム移行プロジェクトに関する知識の引き継ぎ",
  "tags": ["プロジェクトA", "システム移行", "2024"]
}
```

**レスポンス** `201 Created`

```json
{
  "id": "sess_abc123",
  "topic": "プロジェクトAの引き継ぎ",
  "status": "active",
  "created_at": "2025-03-17T10:00:00Z",
  "facts_count": 0,
  "questions_generated": 0
}
```

### セッション一覧を取得

```
GET /sessions
```

**クエリパラメータ**

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `status` | string | `all` | `active`, `completed`, `archived` |
| `limit` | int | 20 | 取得件数（最大100） |
| `offset` | int | 0 | オフセット |

### セッション詳細を取得

```
GET /sessions/{session_id}
```

---

## Documents（ドキュメント管理）

### ドキュメントをアップロード

```
POST /sessions/{session_id}/documents
Content-Type: multipart/form-data
```

**パラメータ**

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `file` | file | Yes | PDF または テキストファイル |
| `description` | string | No | ドキュメントの説明 |

**レスポンス** `202 Accepted`

```json
{
  "id": "doc_xyz789",
  "filename": "handover_notes.pdf",
  "status": "processing",
  "pages": 12,
  "estimated_facts": null,
  "processing_started_at": "2025-03-17T10:01:00Z"
}
```

ドキュメントの処理は非同期で行われます。ステータスは `GET /documents/{document_id}` で確認できます。

### ドキュメントの処理状態を確認

```
GET /sessions/{session_id}/documents/{document_id}
```

**レスポンス**

```json
{
  "id": "doc_xyz789",
  "status": "completed",
  "facts_extracted": 47,
  "inconsistencies_found": 3,
  "processing_completed_at": "2025-03-17T10:02:30Z"
}
```

---

## Questions（質問管理）

### 次の質問を取得

```
GET /sessions/{session_id}/questions/next
```

ELが現在の知識状態に基づいて、最も優先度の高い質問を返します。

**レスポンス** `200 OK`

```json
{
  "id": "q_001",
  "text": "引き継ぎ資料では3月にシステム移行が完了したとありますが、移行後の検証フェーズについての記載がありません。検証はどのように実施されましたか？",
  "type": "missing_information",
  "priority": "high",
  "context": {
    "related_facts": ["fact_012", "fact_015"],
    "source_document": "doc_xyz789",
    "detected_gap": "移行完了後の検証プロセスに関する情報が欠落"
  }
}
```

**質問タイプ**

| type | 説明 |
|------|------|
| `missing_information` | 不足情報の補完を求める質問 |
| `inconsistency` | 矛盾の解消を求める質問 |
| `clarification` | 曖昧な情報の明確化を求める質問 |
| `deep_dive` | 既知の情報に対する深掘り質問 |

### 質問に回答

```
POST /sessions/{session_id}/responses
```

**リクエスト**

```json
{
  "question_id": "q_001",
  "response_text": "検証は4月に2週間かけて実施しました。テスト項目は全200件で、うち5件の軽微な不具合が見つかりましたが、4月末までにすべて修正完了しています。",
  "confidence": "high"
}
```

**レスポンス** `200 OK`

```json
{
  "facts_extracted": 4,
  "new_facts": [
    {
      "id": "fact_048",
      "content": "システム移行の検証は4月に2週間で実施",
      "category": "process",
      "tags": ["検証", "移行", "4月"]
    }
  ],
  "follow_up_available": true
}
```

### 質問をスキップ

```
POST /sessions/{session_id}/questions/{question_id}/skip
```

---

## Facts（ファクト管理）

### ファクト一覧を取得

```
GET /sessions/{session_id}/facts
```

**クエリパラメータ**

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `category` | string | `all` | ファクトのカテゴリで絞り込み |
| `tag` | string | - | タグで絞り込み（複数可） |
| `search` | string | - | 全文検索 |

### ファクトを修正

```
PUT /sessions/{session_id}/facts/{fact_id}
```

### ファクトを削除

```
DELETE /sessions/{session_id}/facts/{fact_id}
```

---

## Knowledge Map（ナレッジマップ）

### ナレッジマップデータを取得

```
GET /sessions/{session_id}/knowledge-map
```

**レスポンス**

```json
{
  "nodes": [
    {
      "id": "fact_001",
      "label": "システム移行完了（3月）",
      "importance": 0.85,
      "category": "milestone",
      "x": 120.5,
      "y": 80.3
    }
  ],
  "edges": [
    {
      "source": "fact_001",
      "target": "fact_048",
      "relationship": "followed_by",
      "weight": 0.72
    }
  ],
  "clusters": [
    {
      "id": "cluster_001",
      "label": "システム移行フェーズ",
      "node_ids": ["fact_001", "fact_012", "fact_048"]
    }
  ]
}
```

---

## Summary（サマリー）

### ナレッジサマリーを取得

```
GET /sessions/{session_id}/summary
```

蓄積された知識の概要レポートを生成します。

**レスポンス**

```json
{
  "total_facts": 47,
  "coverage_score": 0.73,
  "key_findings": [
    "システム移行は計画通り3月に完了し、4月の検証も問題なく終了",
    "運用チームへの引き継ぎドキュメントは作成済み"
  ],
  "remaining_gaps": [
    "障害発生時のエスカレーションフローが未定義",
    "バックアップリストアの手順が未検証"
  ],
  "inconsistencies": [
    {
      "description": "移行日程について資料Aでは3月1日、資料Bでは3月15日と記載",
      "status": "unresolved"
    }
  ]
}
```

---

## Webhooks

処理完了やイベント発生時にWebhookで通知を受け取れます。

```
POST /webhooks
```

**リクエスト**

```json
{
  "url": "https://your-app.example.com/el-webhook",
  "events": ["document.processed", "inconsistency.detected", "session.completed"]
}
```

**イベント一覧**

| イベント | 説明 |
|---------|------|
| `document.processed` | ドキュメント処理完了 |
| `inconsistency.detected` | 新しい矛盾を検知 |
| `question.generated` | 新しい質問が生成された |
| `session.completed` | セッションが完了 |

---

## エラーレスポンス

すべてのエラーは以下の形式で返されます。

```json
{
  "error": {
    "code": "invalid_license",
    "message": "ライセンスキーが無効です",
    "details": null
  }
}
```

| HTTPステータス | コード | 説明 |
|---------------|--------|------|
| 401 | `invalid_license` | ライセンスキーが無効 |
| 403 | `license_expired` | ライセンス期限切れ |
| 404 | `not_found` | リソースが見つからない |
| 422 | `validation_error` | リクエストパラメータ不正 |
| 429 | `rate_limited` | レート制限超過 |
| 503 | `llm_unavailable` | LLM APIに接続不可 |

---

## レート制限

| プラン | リクエスト/分 | セッション数 | ファクト上限/セッション |
|--------|-------------|-------------|----------------------|
| Standard | 60 | 50 | 1,000 |
| Professional | 300 | 200 | 5,000 |
| Enterprise | カスタム | 無制限 | 無制限 |
