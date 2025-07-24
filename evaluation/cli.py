import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import os
import click
import pyjson5

from evaluation.client.lean_client import Lean4Client
from evaluation.client.llm_client import LLMClient
from evaluation.evaluator.basic_evaluator import BasicEvaluator
from evaluation.util import logger


# Common CLI decorators
def common_options(func):
    """Common CLI options shared across most commands."""
    func = click.option(
        "-c",
        "--config",
        type=click.Path(exists=True),
        default="./config/template.json5",
        help="Path to the configuration file (default: ./config/template.json5)",
    )(func)
    func = click.option(
        "-o", "--output-dir", type=click.Path(), default="./eval_logs",
        help="Directory to save evaluation logs and results (default: ./eval_logs)"
    )(func)
    func = click.option(
        "-m", "--model-name", type=str, default=None,
        help="Name of the model to use (overrides config if provided)"
    )(func)
    func = click.option(
        "-n", "--n", type=int, default=None,
        help="Number of samples to generate per problem (overrides config if provided)"
    )(func)
    func = click.option(
        "-t", "--temperature", type=float, default=None,
        help="Sampling temperature for the LLM (overrides config if provided)"
    )(func)
    func = click.option(
        "-tks", "--max-tokens", type=int, default=None,
        help="Maximum number of tokens to generate per sample (overrides config if provided)"
    )(func)
    func = click.option(
        "-g",
        "--greedy-mode",
        is_flag=True,
        default=False,
        help="Enable greedy mode: stop generating/verifying after first success for each problem",
    )(func)
    return func


def setup_output_directory(
    output_dir: Path,
    evaluator: BasicEvaluator,
    suffix: str = "",
    extra_suffixes: dict = None,
    method_name: str = "",
) -> Path:
    """Setup and return the output directory with appropriate naming."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = evaluator.llm_name.split("/")[-1]
    dataset_name = evaluator.dataset_name.split("/")[-1]

    # Build suffixes
    suffix_parts = []
    if method_name:
        suffix_parts.append(method_name)
    if getattr(evaluator, "greedy_mode", False):
        suffix_parts.append("greedy")
    if extra_suffixes:
        suffix_parts.extend([f"{k}-{v}" for k, v in extra_suffixes.items()])

    full_suffix = f"-{'.'.join(suffix_parts)}" if suffix_parts else ""

    final_dir = (
        output_dir
        / f"{model_name}-{dataset_name}-pass@{evaluator.n}-t{evaluator.temperature}-tks{evaluator.max_tokens}{suffix}{full_suffix}"
    )
    final_dir.mkdir(parents=True, exist_ok=True)

    return final_dir


def create_clients(evaluator: BasicEvaluator) -> tuple[LLMClient, Lean4Client]:
    """Create and return LLM and Lean clients."""
    llm_client = LLMClient(
        llm_client_type=evaluator.llm_client_type,
        llm_name=evaluator.llm_name,
        llm_server_url=evaluator.llm_server_url,
        llm_server_api_key=evaluator.llm_server_api_key,
        system_prompt=evaluator.system_prompt,
        temperature=evaluator.temperature,
        max_tokens=evaluator.max_tokens,
        timeout=evaluator.llm_server_timeout,
    )

    lean_client = Lean4Client(
        url=evaluator.lean_server_url,
        api_key=evaluator.lean_server_api_key,
    )

    return llm_client, lean_client


def save_results(
    records: dict, output_dir: Path, evaluator: BasicEvaluator, extra_stats: dict = None
) -> None:
    """Save evaluation results and statistics."""
    from evaluation.constant import ErrorType

    result_file = (
        output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with open(result_file, "w") as f:
        for result in records.values():
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    logger.info(f"Results saved to {result_file}")

    # Save pass@n, error types and one success code each if any
    statistics_file = output_dir / "statistics.json"

    error_types = {}
    for record in records.values():
        for err_type in record.error_types:
            error_types[err_type.name] = error_types.get(err_type.name, 0) + 1
    logger.info(f"Error Types: {error_types}")

    pass_at_ns = {}
    i = 0

    while 2**i <= evaluator.n:
        n = 2**i
        is_pass = []
        for record in records.values():
            if not record.error_types:
                is_pass.append(False)
                continue

            # Take first n error types and check if any is SUCCESS
            is_pass.append(
                any(
                    err_type == ErrorType.SUCCESS for err_type in record.error_types[:n]
                )
            )
        pass_at_n = sum(is_pass) / len(is_pass)
        pass_at_ns[f"pass@{n}"] = pass_at_n
        logger.info(f"Pass@{n}: {pass_at_n * 100:.2f}%")
        i += 1

    one_success_codes = {}
    for record in records.values():
        if record.one_success_solution is not None:
            one_success_codes[record.problem_id] = record.one_success_solution

    stats_data = {
        "pass_at_n": pass_at_ns,
        "error_types": error_types,
        "success_cases": one_success_codes,
        "greedy_mode": getattr(evaluator, "greedy_mode", False),
    }

    if extra_stats:
        stats_data.update(extra_stats)

    with open(statistics_file, "w") as f:
        json.dump(stats_data, f, indent=4, ensure_ascii=False)
    logger.info(f"statistics saved to {statistics_file}")


def evaluate_from_cli(
    evaluator_class: type[BasicEvaluator],
    config_path: str,
    output_dir: str,
    model_name: str = None,
    n: int = None,
    temperature: float = None,
    max_tokens: int = None,
    greedy_mode: bool = False,
    evaluator_kwargs: dict = None,
    suffix: str = "",
    extra_suffixes: dict = None,
    method_name: str = "",
) -> dict:
    """Unified CLI evaluation function for all evaluators."""

    # Load configuration
    config_data = pyjson5.load(open(config_path))
    if evaluator_kwargs:
        config_data.update(evaluator_kwargs)

    # Create evaluator
    evaluator = evaluator_class(**config_data)

    # Override configuration from CLI
    if model_name is not None:
        evaluator.llm_name = model_name
    if n is not None:
        evaluator.n = n
    if temperature is not None:
        evaluator.temperature = temperature
    if max_tokens is not None:
        evaluator.max_tokens = max_tokens
    if greedy_mode:
        evaluator.greedy_mode = greedy_mode

    # Setup output directory
    output_path = Path(output_dir)
    final_output_dir = setup_output_directory(
        output_path, evaluator, suffix, extra_suffixes, method_name
    )

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = final_output_dir / f"{timestamp}.log"
    logger.add(str(log_file))

    logger.info(f"Output directory: {final_output_dir}")
    if getattr(evaluator, "greedy_mode", False):
        logger.info(
            "Greedy mode enabled: will stop after first success for each problem"
        )

    # Create clients
    llm_client, lean_client = create_clients(evaluator)

    # Run evaluation
    output_file = open(final_output_dir / "real_time_responses.jsonl", "w")

    results = evaluator.run_evaluation(llm_client, lean_client, output_file)
    output_file.close()

    # Prepare extra statistics
    extra_stats = {}
    if hasattr(evaluator, "max_turns"):
        extra_stats["multi_turn"] = {
            "max_turns": evaluator.max_turns,
        }

    # Save results
    save_results(results, final_output_dir, evaluator, extra_stats)

    os._exit(0)


@click.group()
def cli():
    """Easy Verify - Unified CLI for LLM formal mathematics evaluation."""
    pass


@cli.command()
@common_options
def online_one_stage(
    config, output_dir, model_name, n, temperature, max_tokens, greedy_mode
):
    """Run online one-stage evaluation."""
    from evaluation.evaluator.online_one_stage import OnlineOneStageEvaluator

    evaluate_from_cli(
        OnlineOneStageEvaluator,
        config,
        output_dir,
        model_name,
        n,
        temperature,
        max_tokens,
        greedy_mode,
        method_name="online-one-stage",
    )


@cli.command()
@common_options
def online_two_stage(
    config, output_dir, model_name, n, temperature, max_tokens, greedy_mode
):
    """Run online two-stage evaluation with proof verification stages."""
    from evaluation.evaluator.online_two_stage import OnlineTwoStageEvaluator

    evaluate_from_cli(
        OnlineTwoStageEvaluator,
        config,
        output_dir,
        model_name,
        n,
        temperature,
        max_tokens,
        greedy_mode,
        method_name="online-two-stage",
    )


if __name__ == "__main__":
    cli()
