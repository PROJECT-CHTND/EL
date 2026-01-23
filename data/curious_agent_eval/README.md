## curious_agent_eval / gold_slots.json 作成ガイド

このディレクトリは、**好奇心駆動インタビューエージェントの評価用テストデータ**をまとめる場所です。  
ここでは、各ケースの「正解データ」である `gold_slots.json` を、誰でも同じルールで作れるようにガイドラインを定義します。

---

### 1. ディレクトリ構造の約束

- 1ケース = 1ディレクトリ として配置します。
- パスの形（推奨）:

```text
data/curious_agent_eval/
  └─ <domain>/                # postmortem | sop | recipe | daily_work
       └─ <case_id>/          # 任意（voicemail_delete, api_timeout_2025 など）
            ├─ gold_slots.json       # 正解スロット（このガイドの対象）
            ├─ system_prompt.txt     # そのケースで使った System プロンプト
            ├─ user_initial_note.txt # 最初にユーザーが話す概要
            ├─ chatgpt_output.json   # 比較対象モデルの出力（任意）
            └─ chatgpt_dialogue.md   # 対話ログ（任意）
```

**このガイドでは、`gold_slots.json` の中身の作り方だけを説明します。**

---

### 2. 作業の流れ（全ドメイン共通）

1. **ドメインを決める**
   - `postmortem` / `sop` / `recipe` / `daily_work` のいずれか。

2. **「理想の完成版」を頭の中で（またはメモで）作る**
   - 人間が読んで「これなら十分」と思えるレベルのポストモーテム/SOP/レシピ/日報をイメージしてください。
   - それを **スロット（項目）ごとに分解したもの**が `gold_slots.json` になります。

3. **ドメインごとのスキーマに従って JSON を書く**
   - 次のセクションで、ドメイン別に「必須フィールド」と「例」を示します。
   - すべて **日本語** で書きます。

4. **「ユーザー初期入力」より少しリッチにする**
   - `user_initial_note.txt` に書かれている内容より、**1〜2段階詳しい情報**（時刻・数値・owner・dueなど）を入れてください。
   - ただし、**現実世界でそのケースが起こったときに無理なくありそうなレベル**に留めてください（盛りすぎない）。

5. **フォーマット**
   - 2スペースまたは4スペースインデントの整形済み JSON。
   - 末尾カンマは入れない。
   - 文字列はすべてダブルクォート `"..."`。

---

### 3. ポストモーテム（`postmortem`）の gold_slots.json

#### 3.1 スキーマ（最低限）

```json
{
  "summary": "string",
  "impact": {
    "users": "string",
    "severity": "string",
    "business_effect": "string"
  },
  "detection": {
    "method": "string",
    "ttd": "string"
  },
  "timeline": [
    { "t": "string", "action": "string", "result": "string" }
  ],
  "root_cause": "string",
  "contributing_factors": ["string"],
  "remediation": ["string"],
  "CAPA": [
    { "owner": "string", "due": "string", "success": "string" }
  ],
  "lessons": "string"
}
```

#### 3.2 記述のポイント

- **summary**: 1文で「いつ・どこで・何が起き・誰に影響したか」。
- **impact.users**: ユーザー種別や規模（例: 「決済APIの利用者全体」「社内の特定部署のみ」）。
- **impact.severity**: できれば、元チケットにある「緊急度」「影響度」を日本語でまとめる。
- **detection**: 検知方法（PagerDuty/問い合わせ/監視など）と、検知までのおおよその時間。
- **timeline**: 3〜7行程度で、主要イベントを「時刻＋アクション＋結果」で書く。
- **root_cause**: 技術/プロセス/組織のどれか、または複合。
- **CAPA.success**: 「何ができていれば成功とみなすか」を定量/定性的に書く。

#### 3.3 例（留守番電話誤削除ケース）

実例は `postmortem/voicemail_delete/gold_slots.json` を参照してください。

---

### 4. 手順書（SOP, `sop`）の gold_slots.json（推奨スキーマ）

`CURIOUS_AGENT_FINAL.md` の 11.1 を簡略化した形です。  
細かい項目をすべて網羅する必要はなく、**重要スロットが埋まっていればOK**です。

#### 4.1 スキーマ（例）

```json
{
  "objective": "string",
  "environment": {
    "scope": "string",
    "env": "string",
    "blast_radius": "string",
    "rollbackable": true
  },
  "prerequisites": ["string"],
  "steps": [
    {
      "id": "string",
      "desc": "string",
      "cmd": "string",
      "expected": "string"
    }
  ],
  "branches": [
    {
      "cond": "string",
      "path": "string"
    }
  ],
  "validation": [
    {
      "obs": "string",
      "pass_criteria": "string"
    }
  ],
  "rollback": [
    {
      "case": "string",
      "steps": ["string"]
    }
  ],
  "hazards": [
    {
      "type": "string",
      "impact": "string",
      "mitigation": "string"
    }
  ]
}
```

#### 4.2 記述のポイント

- **objective**: 「何のための手順か」を1〜2文で。
- **environment**: プロダクション/ステージング、影響範囲、ロールバック可否。
- **steps**: 3〜10ステップ程度。コマンドがない場合は `cmd` を省略してもよい。
- **branches**: 条件分岐があれば1〜3個書く。なければ空配列でもよい。
- **validation**: 「どうなっていれば成功か」を書く（メトリクスや画面状態など）。
- **rollback**: 代表的な失敗パターンと、そのときの戻し方。
- **hazards**: 誤操作やリスク（例: データ削除・トラフィック切替ミスなど）。

---

### 5. レシピ（`recipe`）の gold_slots.json（推奨スキーマ）

#### 5.1 スキーマ（例）

```json
{
  "basic": {
    "dish": "string",
    "servings": "string"
  },
  "ingredients": [
    { "name": "string", "qty": "number or string", "unit": "string" }
  ],
  "tools": ["string"],
  "prep": ["string"],
  "steps": [
    {
      "order": 1,
      "heat": "string",
      "time": "string",
      "temp": "string",
      "desc": "string"
    }
  ],
  "substitutions": ["string"],
  "pitfalls": ["string"],
  "storage": "string"
}
```

#### 5.2 記述のポイント

- **ingredients**: できるだけ「g・ml・大さじ・小さじ」など単位つきで。
- **steps**: 各工程に **火加減（heat）・時間（time）・温度（temp）** をできるだけ入れる。
- **substitutions**: アレルギー・制約に配慮した代替材料。
- **pitfalls**: 焦げやすい・固まりやすいなど、ありがちな失敗。
- **storage**: 冷蔵/冷凍の可否と日数目安。

---

### 6. 業務報告（`daily_work`）の gold_slots.json（推奨スキーマ）

#### 6.1 スキーマ（例）

```json
{
  "subject": "string",
  "projects": ["string"],
  "tasks": [
    {
      "desc": "string",
      "time_spent": "string",
      "link": "string"
    }
  ],
  "artifacts": ["string"],
  "blockers": [
    {
      "desc": "string",
      "cause": "string",
      "mitigation": "string"
    }
  ],
  "next_step": ["string"]
}
```

#### 6.2 記述のポイント

- **subject**: 「2025-11-28 Backend 日報」のように日付＋役割が分かる形がおすすめ。
- **projects**: 関わったプロジェクト名を列挙。
- **tasks.time_spent**: 「2h」「30min」「0.5d」などラフでよいので時間感を入れる。
- **tasks.link**: PR / Issue / ドキュメント URL など（なければ省略可）。
- **blockers**: 本当に困っているものだけでよいが、原因と暫定対処を書いておくと評価しやすい。
- **next_step**: 翌営業日にやる具体アクションを 1〜3 個。

---

### 7. よくある質問（FAQ）

- **Q. すべてのフィールドを必ず埋める必要がありますか？**  
  - A. いいえ。現実に存在しない情報を無理にでっち上げる必要はありません。  
    ただし、「このケースではここが重要だよね」というスロット（CURIOUS_AGENT_FINAL の重要スロット）は、できるだけ埋めてください。

- **Q. 数値（時間・件数など）はどの程度厳密に書くべきですか？**  
  - A. 評価で比較できる程度に一貫していればOKです。  
    例: 「約2時間」「1〜2営業日」「10件程度」などでも問題ありません。

- **Q. 迷ったらどうすればいいですか？**  
  - A. 既存のサンプル（`postmortem/voicemail_delete/gold_slots.json` など）をコピーして、  
    自分のケースに合わせて書き換える形がおすすめです。

---

以上のルールに従って `gold_slots.json` を作れば、誰が作っても **比較しやすく & 自動評価しやすい** テストデータになります。  
不明点があれば、このファイルを更新してルール自体も育てていく想定です。


