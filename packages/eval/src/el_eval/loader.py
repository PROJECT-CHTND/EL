"""Data loader for evaluation test cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from el_eval.schemas import EvalDomain, TestCase

logger = logging.getLogger(__name__)

# Default data directory relative to project root
DEFAULT_DATA_DIR = Path("data/curious_agent_eval")


class DataLoader:
    """Load test cases from the evaluation data directory."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize the data loader.
        
        Args:
            data_dir: Path to the evaluation data directory.
                      Defaults to data/curious_agent_eval/.
        """
        if data_dir is None:
            # Try to find data dir relative to current working directory
            self.data_dir = Path.cwd() / DEFAULT_DATA_DIR
        else:
            self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")

    def _parse_gold_slots(self, content: str) -> dict:
        """Parse gold_slots.json content, handling potential markdown headers."""
        # Some files have markdown headers before JSON
        lines = content.strip().split("\n")
        json_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("{"):
                json_start = i
                break
            elif line.startswith("#"):
                # Skip markdown headers
                continue
        
        json_content = "\n".join(lines[json_start:])
        return json.loads(json_content)

    def _load_case_from_dir(self, case_dir: Path, domain: EvalDomain) -> TestCase | None:
        """Load a single test case from a directory.
        
        Args:
            case_dir: Path to the case directory.
            domain: The domain this case belongs to.
            
        Returns:
            TestCase or None if required files are missing.
        """
        gold_slots_path = case_dir / "gold_slots.json"
        initial_note_path = case_dir / "user_initial_note.txt"
        
        if not gold_slots_path.exists():
            logger.warning(f"Skipping {case_dir}: gold_slots.json not found")
            return None
        
        if not initial_note_path.exists():
            logger.warning(f"Skipping {case_dir}: user_initial_note.txt not found")
            return None
        
        try:
            # Load gold slots
            with open(gold_slots_path, encoding="utf-8") as f:
                gold_slots = self._parse_gold_slots(f.read())
            
            # Load initial note
            with open(initial_note_path, encoding="utf-8") as f:
                initial_note = f.read().strip()
            
            # Load optional system prompt
            system_prompt = None
            system_prompt_path = case_dir / "system_prompt.txt"
            if system_prompt_path.exists():
                with open(system_prompt_path, encoding="utf-8") as f:
                    system_prompt = f.read().strip()
            
            # Load optional reference dialogue
            reference_dialogue = None
            dialogue_path = case_dir / "chatgpt_dialogue.md"
            if dialogue_path.exists():
                with open(dialogue_path, encoding="utf-8") as f:
                    reference_dialogue = f.read()
            
            return TestCase(
                case_id=case_dir.name,
                domain=domain,
                initial_note=initial_note,
                gold_slots=gold_slots,
                system_prompt=system_prompt,
                reference_dialogue=reference_dialogue,
            )
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse gold_slots.json in {case_dir}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load case from {case_dir}: {e}")
            return None

    def _load_standalone_gold_slots(self, json_path: Path, domain: EvalDomain) -> TestCase | None:
        """Load a standalone gold_slots.json file without a directory.
        
        For these files, we need to infer the initial note from the summary.
        """
        try:
            with open(json_path, encoding="utf-8") as f:
                gold_slots = self._parse_gold_slots(f.read())
            
            # Generate case_id from filename
            case_id = json_path.stem.replace("gold_slots_", "").replace("gols_slots_", "")
            
            # Use summary as initial note if available, otherwise create a generic one
            summary = gold_slots.get("summary", "")
            if summary:
                # Create a simplified initial note from summary
                initial_note = self._simplify_summary(summary, domain)
            else:
                initial_note = f"{domain.value}について報告したいです。"
            
            return TestCase(
                case_id=case_id,
                domain=domain,
                initial_note=initial_note,
                gold_slots=gold_slots,
            )
        
        except Exception as e:
            logger.error(f"Failed to load standalone gold_slots from {json_path}: {e}")
            return None

    def _simplify_summary(self, summary: str, domain: EvalDomain) -> str:
        """Create a simplified initial note from a detailed summary.
        
        The initial note should be vague enough to require follow-up questions.
        """
        # Domain-specific simplification
        if domain == EvalDomain.RECIPE:
            # Extract dish name if present
            if "『" in summary and "』" in summary:
                dish = summary.split("『")[1].split("』")[0]
                return f"{dish}の作り方について教えてください。"
            return "料理のレシピについて教えてください。"
        
        elif domain == EvalDomain.POSTMORTEM:
            return "インシデントについて報告したいです。"
        
        elif domain == EvalDomain.MANUAL:
            return "手順書を作成したいです。"
        
        else:
            return "報告したいことがあります。"

    def load_domain(self, domain: EvalDomain) -> list[TestCase]:
        """Load all test cases for a specific domain.
        
        Args:
            domain: The domain to load.
            
        Returns:
            List of TestCase objects.
        """
        domain_dir = self.data_dir / domain.value
        if not domain_dir.exists():
            logger.warning(f"Domain directory not found: {domain_dir}")
            return []
        
        cases: list[TestCase] = []
        
        for item in domain_dir.iterdir():
            if item.is_dir():
                # Directory-based case
                case = self._load_case_from_dir(item, domain)
                if case:
                    cases.append(case)
            elif item.suffix == ".json" and "gold_slots" in item.name:
                # Standalone gold_slots file
                case = self._load_standalone_gold_slots(item, domain)
                if case:
                    cases.append(case)
        
        logger.info(f"Loaded {len(cases)} cases for domain {domain.value}")
        return cases

    def load_all(self) -> list[TestCase]:
        """Load all test cases from all domains.
        
        Returns:
            List of all TestCase objects.
        """
        all_cases: list[TestCase] = []
        
        for domain in EvalDomain:
            cases = self.load_domain(domain)
            all_cases.extend(cases)
        
        logger.info(f"Loaded {len(all_cases)} total test cases")
        return all_cases

    def iter_cases(
        self,
        domain: EvalDomain | None = None,
        case_id: str | None = None,
    ) -> Iterator[TestCase]:
        """Iterate over test cases with optional filtering.
        
        Args:
            domain: Filter by domain (optional).
            case_id: Filter by specific case ID (optional).
            
        Yields:
            TestCase objects matching the filters.
        """
        if domain:
            cases = self.load_domain(domain)
        else:
            cases = self.load_all()
        
        for case in cases:
            if case_id and case.case_id != case_id:
                continue
            yield case

    def get_case(self, case_id: str) -> TestCase | None:
        """Get a specific test case by ID.
        
        Args:
            case_id: The case ID to find.
            
        Returns:
            TestCase or None if not found.
        """
        for case in self.load_all():
            if case.case_id == case_id:
                return case
        return None

    def list_cases(self) -> list[dict[str, str]]:
        """List all available test cases.
        
        Returns:
            List of dicts with case_id and domain.
        """
        return [
            {"case_id": case.case_id, "domain": case.domain.value}
            for case in self.load_all()
        ]
