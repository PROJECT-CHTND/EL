## R4 研究トラック: 方策最適化とLLMモデリング

本ドキュメントは、M1〜M3で安定稼働した好奇心駆動エージェントを、更なる効率・品質へ最適化するための研究計画をまとめる。プロダクト本流（VoI/停止/KPI校正）から責務を分離し、低リスクから段階導入する。

---

### 1. 目的と前提
- 目的: ask/search/none とスロット選択の方策を改善し、ターン数・重複率を下げつつ品質KPIを維持/向上。
- 前提: KPI安定（連続Nセッション）、計測基盤（WAL/トレース/ダッシュボード）、フェイルオーバー方策（VoI）を常備。

---

### 2. 段階構成（R4a→R4d）

#### R4a: 非RLの安全最適化（本流と親和）
- 手法: ベイズ最適化/多腕バンディット/グリッド探索
- 対象: τ_stop, qcheck閾値, importance補正, 温度T
- オンラインA/Bで安全に最適化（LLMは無改変）
 - 重要: すべての実験で**振る舞い方策確率（propensity）**をログし、OPEが可能な状態を維持

#### R4b: オフライン方策改善（OPE/保守的改善）
- 手法: コンテキスト・バンディット、CPI/SPIBB/CQL的発想
- データ: ログ（状態-行動-結果）を用い、IPS/DR/SNIPSでオフライン評価
- 作用点: ask/search/none＋スロット選択の離散方策（Orchestrator上位）

#### R4c: 安全制約付きオンラインRL
- 手法: Safe RL/保守的探索、KL/逸脱率制約で差分実行率を制御
- 作用点: Orchestratorの意思決定（LLMはAPI中心）
- フェイルオーバー: KPI逸脱時に即時VoI方策へ切り戻し

#### R4d: LLM微調整（任意最終段）
- 手法: DPO/GRPO/RLAIF 等
- ガード: JSON厳格/スキーマ適合率/幻覚率にハード制約、本線KPI維持

---

### 3. 報酬設計（研究用）

#### 3.1 ターン報酬/終端報酬の例
```
r_t = w1·Δslot_coverage_t + w2·Δuncertainty_reduction_t
      − w3·question_cost_t − w4·duplicate_penalty_t − w5·latency_penalty_t
R_T = +wQ·final_quality_score
```

#### 3.2 KPI→報酬の写像ルール
- KPIと完全整合（付録B）
- 重みはオフライン最適化で初期化→R4a/bで校正

#### 3.3 ログ設計（OPE必須項目）
```
log_record = {
  ts, user_hash, session_id, policy_id, seed,
  state_hash,                         // Slot充足ベクトル等で安定化
  action: { A: ask|search|none, slot: s | null, bundle: [s1,s2]? },
  policy_prob: {                      // 振る舞い方策の確率（必須）
    p_A: π_b(A|x),
    p_slot: π_b(s|x,A),              // Aがask/searchの場合
    p_bundle: π_b(S|x,A)             // 束ね質問（ある場合）
  },
  q_values_proxy: {...}?,            // 任意：方策スコア/ロジット
  rewards: {                         // 代理報酬は逐次更新
    delta_slot_coverage, delta_uncertainty,
    duplicate_penalty, cost_tokens, latency_sec
  },
  terminal_quality?: final_quality_score,
  stop_reason, voi_value, constraints: {json_valid, halluc_rate, cost_budget}
}
```
注: `policy_prob` と `state_hash` は OPE の必須要件。`policy_id` は全ログに付与。

#### 3.4 DPO/GRPO 用データ作成（R4d）
- ペア生成: 同一状態で（a）既存質問 vs QCheck高スコア、（b）VoI上位 vs 低位、（c）JSON準拠 vs 非準拠
- 弱教師: QCheckスコア＋ルールで自動ラベル、要所は人手校正
- 損失: 相対好み（DPO/GRPO）＋構造違反ペナルティ（関数呼出/JSON schema loss）

---

### 4. オフライン評価（OPE）
- 推奨: IPS, Doubly Robust (DR), Self-Normalized IPS (SNIPS), Switch-DR, WIS
- 有効サンプルサイズ（ESS）と重み分布を監視、低ESSセグメントは結論保留
- 事前に優越が担保できない方策は本番投入しない

#### 4.1 階層行動のOPE（要旨）
- 単発行動: `w = π_e(A|x)/π_b(A|x) × π_e(s|x,A)/π_b(s|x,A)`
- none: `w = π_e(none|x)/π_b(none|x)`
- 束ね質問 S（近似）:
  - 独立仮定: `w ≈ π_e(A|x)/π_b(A|x) × ∏_{s∈S} π_e(s|x,A)/π_b(s|x,A)`
  - 将来拡張: slate bandit（位置/選択）モデルで厳密OPE

---

### 5. オンライン実験設計
- 固定割当（ユーザ/スレッドハッシュ）で交差汚染防止
- MDE（最小検出効果）と必要セッション数を事前算定
- 主要因子: τ_stop, qcheck閾値, softmax温度T, importance補正係数

---

### 6. 安全・フェイルオーバー
- 即時切り戻し条件: KPI下降、VoI分布の異常、QCheck失敗率上昇
- 実行率制限: ベースライン方策との差分をKL/確率ブレンドで制御
- 監査: WALに方策ID/バージョン/パラメータを記録して再現可能に

#### 6.1 制約付き目的の定式化
- 目的: maximize `J(π) = E[∑ r_t]` s.t. `E[c_i(x_t,a_t)] ≤ b_i`
- 代表制約: `c_json_invalid`, `c_halluc`, `c_cost`（トークン/レイテンシ）, `c_dup`（重複率）
- 運用: `b_i` はプロダクトSLOに対応、ダッシュボード監視と1:1対応

---

### 7. 実装ガイド（責務分離）
- PolicyProviderインタフェース
  - heuristic_voi（既定）/ bandit / offline_policy / safe_rl / dpo_adapter
  - 必須I/F: `id()` に加え、確率出力 `π(A|x)`, `π(s|x,A)`（必要に応じ `π(S|x,A)`）
- 切り戻しゲート: KPI/逸脱率/VoI分布の閾値に違反したら即時ベースラインへ

---

### 8. リスクと対策
| リスク | 例 | 緩和 |
| --- | --- | --- |
| 逸脱による品質悪化 | 方策変更で重複増加 | 差分実行率制限、即時切り戻し、OPE先行 |
| 報酬ハッキング | 短期最適で長期品質低下 | 終端品質報酬を併用、KPI監視 |
| データバイアス | 特定ドメイン過多 | 分層評価、重み付け、再収集 |

---

### 9. 参考文献（実務寄り）
- Slate Bandit / Off-policy evaluation surveys（slate OPEの参考）

---

### 10. 研究KPI（Go/No-Go判定）
- OPE妥当性: `|DR − SNIPS| ≤ δ`（セグメント別）、`ESS ≥ E_min`
- 安全制約: `json_invalid ≤ b_json`, `halluc ≤ b_halluc`, `avg_cost ≤ b_cost`
- 実運用改善（R4a→R4c）: ターン数 −10〜20%、重複率 −20%、品質KPI 維持/改善
- LLM微調整（R4d）: QCheck合格率 +X%、JSON適合率 ≥99%、推論コスト −Y%
- Dudík et al., Doubly Robust Policy Evaluation
- Thomas & Brunskill, High Confidence Off-Policy Evaluation
- Schulman et al., Proximal Policy Optimization (安全制約発想の参考)


