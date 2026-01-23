"""CLI entry point for EL evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from el_eval.loader import DataLoader
from el_eval.runner import TestRunner
from el_eval.schemas import EvalDomain, EvalSummary

# Load environment variables
load_dotenv()

console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def print_summary(summary: EvalSummary) -> None:
    """Print evaluation summary using rich."""
    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold green]Evaluation Complete[/bold green]\n"
        f"Total: {summary.total_cases} cases | "
        f"Completed: {summary.completed_cases} | "
        f"Failed: {summary.failed_cases}",
        title="EL Agent Evaluation",
    ))
    
    # Overall metrics
    metrics_table = Table(title="Overall Metrics", show_header=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Avg Slot Coverage", f"{summary.avg_slot_coverage:.1%}")
    metrics_table.add_row("Avg Turn Count", f"{summary.avg_turn_count:.1f}")
    metrics_table.add_row("Avg Turn Efficiency", f"{summary.avg_turn_efficiency:.2f}")
    metrics_table.add_row("Domain Accuracy", f"{summary.domain_accuracy_rate:.1%}")
    metrics_table.add_row("Total Duration", f"{summary.total_duration_seconds:.1f}s")
    
    console.print(metrics_table)
    
    # Question quality metrics (Phase 2)
    if summary.avg_question_quality is not None:
        console.print()
        quality_table = Table(title="Question Quality (LLM-as-Judge)", show_header=True)
        quality_table.add_column("Dimension", style="cyan")
        quality_table.add_column("Score", style="magenta")
        
        quality_table.add_row("Overall Quality", f"{summary.avg_question_quality:.2f}")
        quality_table.add_row("Empathy", f"{summary.avg_empathy:.2f}" if summary.avg_empathy else "-")
        quality_table.add_row("Insight", f"{summary.avg_insight:.2f}" if summary.avg_insight else "-")
        quality_table.add_row("Specificity", f"{summary.avg_specificity:.2f}" if summary.avg_specificity else "-")
        
        console.print(quality_table)
    
    # Per-domain breakdown
    if summary.by_domain:
        console.print()
        domain_table = Table(title="By Domain", show_header=True)
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Cases", style="white")
        domain_table.add_column("Coverage", style="green")
        domain_table.add_column("Turns", style="yellow")
        domain_table.add_column("Efficiency", style="blue")
        
        # Check if any domain has quality metrics
        has_quality = any("avg_quality" in stats for stats in summary.by_domain.values())
        if has_quality:
            domain_table.add_column("Quality", style="magenta")
        
        for domain, stats in summary.by_domain.items():
            row = [
                domain,
                str(int(stats["count"])),
                f"{stats['avg_coverage']:.1%}",
                f"{stats['avg_turns']:.1f}",
                f"{stats['avg_efficiency']:.2f}",
            ]
            if has_quality:
                row.append(f"{stats.get('avg_quality', 0):.2f}" if "avg_quality" in stats else "-")
            domain_table.add_row(*row)
        
        console.print(domain_table)
    
    # Individual results
    if summary.results:
        console.print()
        results_table = Table(title="Individual Results", show_header=True)
        results_table.add_column("Case ID", style="cyan", max_width=30)
        results_table.add_column("Domain", style="white")
        results_table.add_column("Coverage", style="green")
        results_table.add_column("Turns", style="yellow")
        results_table.add_column("Domain OK", style="blue")
        
        # Check if any result has quality metrics
        has_quality = any(r.question_quality is not None for r in summary.results)
        if has_quality:
            results_table.add_column("Quality", style="magenta")
        
        for result in summary.results:
            domain_ok = "[green]✓[/green]" if result.domain_accuracy else "[red]✗[/red]"
            row = [
                result.case_id[:30],
                result.domain.value,
                f"{result.slot_coverage:.1%}",
                str(result.turn_count),
                domain_ok,
            ]
            if has_quality:
                if result.question_quality:
                    row.append(f"{result.question_quality.avg_overall:.2f}")
                else:
                    row.append("-")
            results_table.add_row(*row)
        
        console.print(results_table)


def print_case_detail(summary: EvalSummary, case_id: str) -> None:
    """Print detailed results for a specific case."""
    result = next((r for r in summary.results if r.case_id == case_id), None)
    if not result:
        console.print(f"[red]Case not found: {case_id}[/red]")
        return
    
    # Build header with quality if available
    header_lines = [
        f"[bold]Case: {result.case_id}[/bold]",
        f"Domain: {result.domain.value}",
        f"Coverage: {result.slot_coverage:.1%} | "
        f"Turns: {result.turn_count} | "
        f"Efficiency: {result.turn_efficiency:.2f}",
    ]
    if result.question_quality:
        header_lines.append(
            f"Quality: {result.question_quality.avg_overall:.2f} | "
            f"Empathy: {result.question_quality.avg_empathy:.2f} | "
            f"Insight: {result.question_quality.avg_insight:.2f}"
        )
    
    console.print()
    console.print(Panel.fit("\n".join(header_lines), title="Case Details"))
    
    # Question quality details (Phase 2)
    if result.question_quality and result.question_quality.questions:
        console.print("\n[bold magenta]Question Quality Breakdown:[/bold magenta]")
        quality_table = Table(show_header=True)
        quality_table.add_column("Turn", style="dim")
        quality_table.add_column("Empathy", style="cyan")
        quality_table.add_column("Insight", style="green")
        quality_table.add_column("Specificity", style="yellow")
        quality_table.add_column("Flow", style="blue")
        quality_table.add_column("Overall", style="magenta")
        
        for q in result.question_quality.questions[:10]:  # Limit to 10
            quality_table.add_row(
                str(q.turn_number),
                f"{q.empathy_score:.2f}",
                f"{q.insight_score:.2f}",
                f"{q.specificity_score:.2f}",
                f"{q.flow_score:.2f}",
                f"{q.overall_score:.2f}",
            )
        console.print(quality_table)
    
    # Conversation log
    console.print("\n[bold cyan]Conversation:[/bold cyan]")
    for log in result.conversation_log:
        console.print(f"\n[dim]Turn {log.turn_number}[/dim]")
        console.print(f"[green]User:[/green] {log.user_message[:200]}...")
        console.print(f"[blue]Agent:[/blue] {log.assistant_response[:200]}...")
        if log.insights_saved:
            console.print(f"[yellow]Insights saved: {len(log.insights_saved)}[/yellow]")
    
    # Matched slots
    if result.matched_slots:
        console.print("\n[bold green]Matched Slots:[/bold green]")
        for match in result.matched_slots[:10]:  # Limit to 10
            console.print(f"  • {match.slot_key}: {match.similarity_score:.2f}")
    
    # Unmatched slots
    if result.unmatched_slots:
        console.print("\n[bold red]Unmatched Slots:[/bold red]")
        for slot in result.unmatched_slots[:10]:  # Limit to 10
            console.print(f"  • {slot}")


@click.group()
def cli() -> None:
    """EL Agent Evaluation CLI."""
    pass


@cli.command("run")
@click.option(
    "--domain",
    "-d",
    type=click.Choice(["postmortem", "manual", "recepi", "daily_work"]),
    help="Filter by domain",
)
@click.option(
    "--case",
    "-c",
    type=str,
    help="Run specific case by ID",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output results to JSON file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--fast",
    is_flag=True,
    help="Use fast mode (no LLM for simulator/metrics)",
)
@click.option(
    "--quality",
    "-q",
    is_flag=True,
    help="Enable question quality evaluation (LLM-as-Judge)",
)
@click.option(
    "--list",
    "list_cases",
    is_flag=True,
    help="List available test cases",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    help="Path to evaluation data directory",
)
@click.option(
    "--detail",
    type=str,
    help="Show detailed results for specific case ID",
)
def run_eval(
    domain: str | None,
    case: str | None,
    output: str | None,
    verbose: bool,
    fast: bool,
    quality: bool,
    list_cases: bool,
    data_dir: str | None,
    detail: str | None,
) -> None:
    """Run evaluation tests against the EL Agent.
    
    Examples:
    
        # Run all tests
        el-eval run
        
        # Run only postmortem cases
        el-eval run --domain postmortem
        
        # Run specific case with quality evaluation
        el-eval run --case voicemail_delete --quality
    """
    setup_logging(verbose)
    
    # Initialize data loader
    try:
        loader = DataLoader(data_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Hint: Run from the project root directory or specify --data-dir[/dim]")
        sys.exit(1)
    
    # List cases mode
    if list_cases:
        cases = loader.list_cases()
        table = Table(title="Available Test Cases", show_header=True)
        table.add_column("Case ID", style="cyan")
        table.add_column("Domain", style="green")
        
        for c in cases:
            table.add_row(c["case_id"], c["domain"])
        
        console.print(table)
        console.print(f"\n[dim]Total: {len(cases)} cases[/dim]")
        return
    
    # Convert domain string to enum
    domain_filter = EvalDomain(domain) if domain else None
    
    # Run evaluation
    console.print("[bold]EL Agent Evaluation[/bold]")
    mode_parts = []
    if fast:
        mode_parts.append("Fast (no LLM for simulator/metrics)")
    else:
        mode_parts.append("Full (with LLM)")
    if quality:
        mode_parts.append("+ Quality Eval")
    console.print(f"Mode: {' '.join(mode_parts)}")
    
    if domain_filter:
        console.print(f"Domain filter: {domain_filter.value}")
    if case:
        console.print(f"Case filter: {case}")
    
    async def run() -> EvalSummary:
        runner = TestRunner(
            use_llm_simulator=not fast,
            use_llm_metrics=not fast,
            evaluate_question_quality=quality,
        )
        return await runner.run_all(
            data_loader=loader,
            domain_filter=domain_filter,
            case_filter=case,
        )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running evaluation...", total=None)
        summary = asyncio.run(run())
    
    # Print results
    if detail:
        print_case_detail(summary, detail)
    else:
        print_summary(summary)
    
    # Save to file if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            # Convert to dict for JSON serialization
            data: dict[str, Any] = {
                "total_cases": summary.total_cases,
                "completed_cases": summary.completed_cases,
                "failed_cases": summary.failed_cases,
                "avg_slot_coverage": summary.avg_slot_coverage,
                "avg_turn_count": summary.avg_turn_count,
                "avg_turn_efficiency": summary.avg_turn_efficiency,
                "domain_accuracy_rate": summary.domain_accuracy_rate,
                "by_domain": summary.by_domain,
                "total_duration_seconds": summary.total_duration_seconds,
            }
            
            # Add quality metrics if available
            if summary.avg_question_quality is not None:
                data["avg_question_quality"] = summary.avg_question_quality
                data["avg_empathy"] = summary.avg_empathy
                data["avg_insight"] = summary.avg_insight
                data["avg_specificity"] = summary.avg_specificity
            
            # Add results
            results_list = []
            for r in summary.results:
                result_dict: dict[str, Any] = {
                    "case_id": r.case_id,
                    "domain": r.domain.value,
                    "slot_coverage": r.slot_coverage,
                    "turn_count": r.turn_count,
                    "turn_efficiency": r.turn_efficiency,
                    "domain_accuracy": r.domain_accuracy,
                    "insights_saved": r.insights_saved,
                }
                if r.question_quality:
                    result_dict["question_quality"] = {
                        "avg_overall": r.question_quality.avg_overall,
                        "avg_empathy": r.question_quality.avg_empathy,
                        "avg_insight": r.question_quality.avg_insight,
                        "avg_specificity": r.question_quality.avg_specificity,
                        "avg_flow": r.question_quality.avg_flow,
                    }
                results_list.append(result_dict)
            data["results"] = results_list
            
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]Results saved to: {output_path}[/dim]")


@cli.command("compare")
@click.option(
    "--case",
    "-c",
    type=str,
    required=True,
    help="Case ID to compare",
)
@click.option(
    "--turns",
    "-t",
    type=int,
    default=5,
    help="Maximum turns per agent",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    help="Path to evaluation data directory",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output results to JSON file",
)
def compare_agents(
    case: str,
    turns: int,
    data_dir: str | None,
    verbose: bool,
    output: str | None,
) -> None:
    """Compare EL Agent vs baseline agents.
    
    Runs the same case through EL Agent and simple baseline agents,
    then compares question quality using LLM-as-Judge.
    
    Examples:
    
        # Compare on a specific case
        el-eval compare --case voicemail_delete
        
        # With more turns
        el-eval compare --case voicemail_delete --turns 8
    """
    setup_logging(verbose)
    
    # Import here to avoid circular imports
    from el_eval.baseline import ComparisonRunner
    
    # Initialize data loader
    try:
        loader = DataLoader(data_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Hint: Run from the project root directory or specify --data-dir[/dim]")
        sys.exit(1)
    
    # Find the case
    test_case = loader.get_case(case)
    if not test_case:
        console.print(f"[red]Case not found: {case}[/red]")
        console.print("[dim]Use 'el-eval run --list' to see available cases[/dim]")
        sys.exit(1)
    
    console.print("[bold]EL Agent vs Baseline Comparison[/bold]")
    console.print(f"Case: {test_case.case_id} (domain: {test_case.domain.value})")
    console.print(f"Max turns: {turns}")
    console.print()
    
    async def run_comparison() -> dict[str, Any]:
        runner = ComparisonRunner()
        return await runner.compare_single_case(
            topic=test_case.initial_note,
            gold_slots=test_case.gold_slots,
            max_turns=turns,
        )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running comparison...", total=None)
        results = asyncio.run(run_comparison())
    
    # Display results
    console.print()
    
    # Create comparison table
    table = Table(title="Quality Comparison", show_header=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Empathy", style="green")
    table.add_column("Insight", style="yellow")
    table.add_column("Specificity", style="blue")
    table.add_column("Flow", style="magenta")
    table.add_column("Overall", style="bold white")
    
    display_names = {
        "el_agent": "EL Agent",
        "baseline_simple": "Simple Baseline",
        "baseline_form": "Form Filler",
    }
    
    for agent_name, scores in results.items():
        table.add_row(
            display_names.get(agent_name, agent_name),
            f"{scores['avg_empathy']:.2f}",
            f"{scores['avg_insight']:.2f}",
            f"{scores['avg_specificity']:.2f}",
            f"{scores['avg_flow']:.2f}",
            f"{scores['avg_overall']:.2f}",
        )
    
    console.print(table)
    
    # Determine winner
    el_score = results.get("el_agent", {}).get("avg_overall", 0)
    baseline_scores = [
        results.get(k, {}).get("avg_overall", 0)
        for k in ["baseline_simple", "baseline_form"]
    ]
    best_baseline = max(baseline_scores) if baseline_scores else 0
    
    console.print()
    if el_score > best_baseline:
        diff = el_score - best_baseline
        console.print(f"[bold green]✓ EL Agent wins by {diff:.2f} points[/bold green]")
    elif el_score < best_baseline:
        diff = best_baseline - el_score
        console.print(f"[bold red]✗ Baseline wins by {diff:.2f} points[/bold red]")
    else:
        console.print("[bold yellow]= Tie[/bold yellow]")
    
    # Save to file if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]Results saved to: {output_path}[/dim]")


def main() -> None:
    """Main entry point - run the CLI."""
    # For backwards compatibility, if no subcommand is provided,
    # default to 'run' behavior
    import sys
    
    # Check if any subcommand is being used
    if len(sys.argv) > 1 and sys.argv[1] in ["run", "compare", "--help", "-h"]:
        cli()
    else:
        # Insert 'run' as the default subcommand
        sys.argv.insert(1, "run")
        cli()


if __name__ == "__main__":
    main()
