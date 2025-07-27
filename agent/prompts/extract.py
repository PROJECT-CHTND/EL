"""Prompt strings for Stage 02 extraction."""

SYSTEM_PROMPT: str = "You are a structured-data extractor."

# Minimal JSON schema hint for the LLM to adhere to
SCHEMA_SNIPPET: str = '{"entities":[], "relations":[]}' 