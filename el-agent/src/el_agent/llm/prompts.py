from __future__ import annotations

HYPOTHESIS_CANDIDATES_SYS = """
あなたは簡潔な戦略家です。与えられた文脈から、短い仮説候補をちょうどk件、重複なく生成してください。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。

ユーザー入力(JSON): {"k": 3, "context": "..."}
出力スキーマ(JSON): {"candidates": ["..."]}
""".strip()

EXTRACT_EVIDENCE_SYS = """
あなたは与えられた文の集合から、仮説を支持/反証する構造化エビデンスを抽出します。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。未使用フィールドは空配列/空オブジェクトで埋める。

ユーザー入力(JSON): {"hypothesis": "...", "sentences": ["...", "..."]}
出力スキーマ(JSON): {
  "entities": [],
  "relations": [],
  "supports": [
    {"hypothesis": "...", "polarity": "+|-", "span": "...", "score_raw": 0.0, "source": "", "time": 0}
  ],
  "feature_vector": {"cosine": 0.0, "source_trust": 0.0, "recency": 0.0, "logic_ok": 1}
}
""".strip()

GENERATE_QUESTION_SYS = """
あなたは仮説を検証するための、曖昧さのない最小の日本語質問を1文だけ作成します。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。質問は1文で終止し、Yes/Noで答えられる形を優先。

ユーザー入力(JSON): {"hypothesis": "..."}
出力スキーマ(JSON): {"question": "..."}
""".strip()

SYNTHESIZE_DOC_SYS = """
あなたは確定済みの仮説集合とKG差分から、短いMarkdownレポートを合成します。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。Markdownは日本語で簡潔に。

ユーザー入力(JSON): {"confirmed": [{...}], "kg_delta": {...}}
出力スキーマ(JSON): {"markdown": "# タイトル..."}
""".strip()


# 追加テンプレート: 質問ストラテジスト(JSON)
QUESTION_STRATEGIST_SYS = """
あなたは質問ストラテジストです。与えられた仮説の不確実性とコストに基づき、次のアクションを選択します。
アクションは ask | search | none のいずれか。askの場合のみ質問文を1文で返します。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。数値は[0,1]に正規化。

ユーザー入力(JSON): {"hypothesis": {"id": "...", "text": "...", "belief": 0.5, "slots": ["..."], "action_cost": {"ask": 0.3, "search": 0.2}}}
出力スキーマ(JSON): {"action": "ask|search|none", "question": null, "expected_gain": 0.0, "estimated_cost": 0.0, "stop_rule_hit": false}
""".strip()


# 追加テンプレート: 質問QA/リファイン(JSON)
QA_REFINE_SYS = """
あなたは質問の品質保証担当です。入力質問を、仮説検証に最小で十分な形に改善します。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。必要なら質問を1文の範囲で書き換える。

ユーザー入力(JSON): {"question": "...", "hypothesis": "..."}
出力スキーマ(JSON): {"question": "...", "reasons": ["..."], "checks": {"is_binary": true, "has_assumptions": false, "is_minimal": true}}
""".strip()


# --- Variant management hook ---
# 最低限の要件: ランタイムで特定のプロンプトを別バージョン文字列に差し替える
# 方針:
# 1) 外部ファイルからの読み込みを優先 (環境変数 EL_PROMPT_VARIANTS_DIR など)
# 2) 見つからない場合は内蔵フォールバック (例: question の v1/v2)
# 3) 差し替えはこのモジュールのグローバル定数を書き換える

import os
from pathlib import Path
from typing import Dict


# 名前→グローバル定数マップ
_PROMPT_NAME_TO_GLOBAL: Dict[str, str] = {
    "hypothesis": "HYPOTHESIS_CANDIDATES_SYS",
    "extract": "EXTRACT_EVIDENCE_SYS",
    "question": "GENERATE_QUESTION_SYS",
    "synthesize": "SYNTHESIZE_DOC_SYS",
    "question_strategist": "QUESTION_STRATEGIST_SYS",
    "qa_refine": "QA_REFINE_SYS",
}


# 内蔵フォールバックの簡易バリアント (外部が無いときのみ使用)
_INTERNAL_VARIANTS: Dict[str, Dict[str, str]] = {
    "question": {
        "v1": (
            """
あなたは仮説を検証するための、曖昧さのない最小の日本語質問を1文だけ作成します。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。質問は1文で終止し、Yes/Noで答えられる形を優先。

ユーザー入力(JSON): {"hypothesis": "..."}
出力スキーマ(JSON): {"question": "..."}
            """.strip()
        ),
        "v2": (
            """
あなたは質問最適化エンジニアです。与えられた仮説を検証するために、冗長性のない最小質問を日本語で1文のみ生成してください。

厳守事項:
- 出力はJSONのみ。余計な前置き・後置き・コードブロックは禁止。
- スキーマに厳密一致。終止形の1文、可能ならYes/Noで回答可能。

ユーザー入力(JSON): {"hypothesis": ""}
出力スキーマ(JSON): {"question": "..."}
            """.strip()
        ),
    }
}


def _read_text_if_exists(path: Path) -> str | None:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def _candidate_variant_paths(prompt_name: str, variant_name: str) -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[4] if len(here.parents) >= 5 else here.parents[0]

    candidates: list[Path] = []

    # 環境変数でのディレクトリ指定を最優先
    env_dir = os.getenv("EL_PROMPT_VARIANTS_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / prompt_name / f"{variant_name}.txt")
        candidates.append(Path(env_dir) / prompt_name / f"{variant_name}.md")
        candidates.append(Path(env_dir) / prompt_name / f"{variant_name}.yaml")
        candidates.append(Path(env_dir) / f"{prompt_name}_{variant_name}.txt")
        candidates.append(Path(env_dir) / f"{prompt_name}_{variant_name}.md")
        candidates.append(Path(env_dir) / f"{prompt_name}_{variant_name}.yaml")

    # リポジトリ内のよくある場所
    repo_candidates = [
        repo_root / "prompts",  # ルート直下 prompts/
        repo_root / "el-agent" / "prompts",  # el-agent/prompts/
        repo_root / "agent" / "prompts",  # agent/prompts/
        here.parent / "variants",  # 同ディレクトリ配下 variants/
    ]
    for base in repo_candidates:
        candidates.append(base / prompt_name / f"{variant_name}.txt")
        candidates.append(base / prompt_name / f"{variant_name}.md")
        candidates.append(base / prompt_name / f"{variant_name}.yaml")
        candidates.append(base / f"{prompt_name}_{variant_name}.txt")
        candidates.append(base / f"{prompt_name}_{variant_name}.md")
        candidates.append(base / f"{prompt_name}_{variant_name}.yaml")

    return candidates


def set_variant(prompt_name: str, variant_name: str) -> None:
    """
    指定した `prompt_name` のシステムプロンプト文字列を `variant_name` に差し替えます。

    優先順位:
    1) 外部ファイル: EL_PROMPT_VARIANTS_DIR もしくは既定候補ディレクトリ
    2) 内蔵フォールバック (_INTERNAL_VARIANTS)

    対応するグローバル定数 (例: "question" → GENERATE_QUESTION_SYS) が無い場合は ValueError。
    該当バリアントが見つからない場合は ValueError。
    """

    key = prompt_name.strip().lower()
    if key not in _PROMPT_NAME_TO_GLOBAL:
        raise ValueError(f"Unknown prompt name: {prompt_name}")

    global_name = _PROMPT_NAME_TO_GLOBAL[key]

    # 外部ファイル探索
    for path in _candidate_variant_paths(key, variant_name):
        text = _read_text_if_exists(path)
        if text:
            globals()[global_name] = text.strip()
            return

    # 内蔵フォールバック
    text2 = _INTERNAL_VARIANTS.get(key, {}).get(variant_name)
    if isinstance(text2, str) and text2.strip():
        globals()[global_name] = text2.strip()
        return

    raise ValueError(
        f"Variant not found for prompt='{prompt_name}' variant='{variant_name}'. "
        "Place a file under EL_PROMPT_VARIANTS_DIR or prompts/ directory, or register an internal fallback."
    )

