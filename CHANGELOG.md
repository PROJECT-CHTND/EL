# Changelog

All notable changes to EL (Eager Learner) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2025-XX-XX

### 🎉 Initial Release

- **Interview Engine**: 質問駆動型のナレッジ抽出エンジン
  - 矛盾検知による自動質問生成
  - 不足情報の検知と深掘り質問
  - セッション管理とコンテキスト保持

- **Fact Ledger**: 構造化された知識管理
  - ファクトの自動抽出・分類・タグ付け
  - ファクト間の関連性分析
  - 検索・フィルタリング機能

- **Document Pipeline**: ドキュメント自動処理
  - PDF・テキストファイルの取り込み
  - 自動チャンク分割・ファクト抽出
  - 既存知識との矛盾チェック

- **Knowledge Map**: ナレッジの可視化
  - ファクトの重要度・関連性をマッピング
  - クラスタリングによるテーマ分類

- **API**: RESTful API（OpenAPI準拠）
  - セッション管理
  - ドキュメントアップロード
  - ファクト管理
  - Webhook通知

- **Python SDK**: `el-sdk` パッケージ
- **Demo Frontend**: 参考UI実装
