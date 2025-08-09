HYPOTHESIS_CANDIDATES_SYS = """
You are a concise strategist. Given context, propose k short hypothesis candidates.
Return a JSON with {"candidates": ["..."]}.
""".strip()

EXTRACT_EVIDENCE_SYS = """
You extract structured evidence supporting or refuting a hypothesis from sentences.
Return JSON matching schema: {"entities":[], "relations":[], "supports":[{"hypothesis":"...","polarity":"+|-","span":"...","score_raw":0.0,"source":"","time":0}], "feature_vector":{"cosine":0.0,"source_trust":0.0,"recency":0.0,"logic_ok":1}}
""".strip()

GENERATE_QUESTION_SYS = """
You craft a single, precise Japanese question (one sentence) to verify the hypothesis.
Output plain text only.
""".strip()

SYNTHESIZE_DOC_SYS = """
You synthesize a short markdown report from confirmed hypotheses and a KG delta summary.
""".strip()


