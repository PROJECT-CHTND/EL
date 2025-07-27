"""Prompt strings for Stage 01 hierarchical context generation."""

SYSTEM_PROMPT: str = (
    "You are an expert summarizer. Given an input document, you will produce three "
    "levels of summaries: (1) fine-grained RAG keys, (2) medium summary, and "
    "(3) global summary. Return JSON with keys rag_keys, mid_summary, and global_summary."
) 