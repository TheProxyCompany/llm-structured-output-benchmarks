import json
import os
import pickle
from typing import Any

import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from tqdm import tqdm
import itertools
import numpy as np

from src.frameworks import factory

app = typer.Typer()


def save_results(results: dict[str, Any], task: str, config_key: str) -> None:
    """Save benchmark results to disk.

    Args:
        results: Dictionary containing benchmark results
        task: Name of the task (determines subdirectory)
        config_key: Name of the framework (determines filename)
    """
    directory = f"results/{task}"
    os.makedirs(directory, exist_ok=True)

    output_path = f"{directory}/{config_key}.pkl"
    with open(output_path, "wb") as file:
        pickle.dump(results, file)
        logger.info(f"Results saved to {output_path}")


def run_framework_evaluation(
    framework_instance: Any,
    task: str,
    n_runs: int,
    row: Any | None = None,  # Can be pandas Series or namedtuple
) -> dict[str, list[Any]]:
    """Run evaluation for a single framework instance.

    Args:
        framework_instance: The framework to evaluate
        task: Name of the task being performed
        n_runs: Number of runs per evaluation
        row: Optional row of test data (can be pandas Series or namedtuple)

    Returns:
        Dictionary containing evaluation results
    """
    run_results = {
        "predictions": [],
        "percent_successful": [],
        "metrics": [],
        "latencies": [],
    }

    try:
        if row is not None:
            if task == "function_calling":
                # Handle function calling data format
                tools = json.loads(row.tools)
                tool_schemas = [
                    {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "const",
                                "const": tool["function"]["name"],
                            },
                            "arguments": tool["function"]["parameters"],
                        },
                        "required": ["name", "arguments"],
                    }
                    for tool in tools
                ]
                inputs = {
                    "prompt": row.prompt,  # List of message dicts
                    "schema": {"anyOf": tool_schemas},
                }
                expected = json.loads(row.completion.split("<functioncall>")[1])
                predictions, percent_successful, framework_metrics, latencies = (
                    framework_instance.run(
                        task=task,
                        n_runs=n_runs,
                        expected_response=expected,
                        inputs=inputs,
                    )
                )
            else:
                # Handle other tasks' labeled data
                labels = set(row.labels) if isinstance(row.labels, list) else row.labels
                predictions, percent_successful, framework_metrics, latencies = (
                    framework_instance.run(
                        task=task,
                        n_runs=n_runs,
                        expected_response=labels,
                        inputs={"text": row.text},
                    )
                )

            if framework_metrics is not None:
                run_results["metrics"].append(framework_metrics)

        else:
            # Handle unlabeled/synthetic data case
            predictions, percent_successful, _, latencies = framework_instance.run(
                task=task, n_runs=n_runs, expected_response=None
            )

        run_results["predictions"].append(predictions)
        run_results["percent_successful"].append(percent_successful)
        run_results["latencies"].append(latencies)

    except Exception as e:
        logger.error(f"Error during framework evaluation: {str(e)}")
        logger.exception(e)

    return run_results


@app.command()
def run_benchmark(config_path: str = "config.yaml"):
    """Run benchmarks for all frameworks specified in the config file.

    Args:
        config_path: Path to YAML config file
    """
    device = "cuda" if torch.cuda.is_available() else "auto"
    logger.info(f"Using device: {device} for local models")

    # Load and validate config
    try:
        with open(config_path, "r") as file:
            configs = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise typer.Exit(1)

    # Run benchmarks for each framework
    for config_key, config_values in configs.items():
        results = {config_key: {}}

        for config in config_values:
            task = config["task"]
            n_runs = config["n_runs"]

            # Initialize framework
            try:
                framework_instance = factory(
                    config_key, task=task, device=device, **config["init_kwargs"]
                )
                logger.info(f"Initialized {type(framework_instance).__name__}")

            except Exception as e:
                logger.error(f"Failed to initialize {config_key}: {str(e)}")
                continue

            # Run evaluation
            if isinstance(framework_instance.source_data, pd.DataFrame):
                framework_results = {
                    "predictions": [],
                    "percent_successful": [],
                    "metrics": [],
                    "latencies": [],
                }

                for row in tqdm(
                    framework_instance.source_data.itertuples(),
                    desc=f"Running {framework_instance.task}",
                    total=len(framework_instance.source_data),
                ):
                    run_results = run_framework_evaluation(
                        framework_instance=framework_instance,
                        task=task,
                        n_runs=n_runs,
                        row=row,
                    )
                    # Accumulate results
                    for key in framework_results:
                        framework_results[key].extend(run_results[key])

                results[config_key] = framework_results
            else:
                run_results = run_framework_evaluation(
                    framework_instance=framework_instance, task=task, n_runs=n_runs
                )
                results[config_key] = run_results

            # Save results after each framework evaluation
            save_results(results, task, config_key)


def fraction_to_percentage(fraction_str: str) -> str:
    """Convert a fraction string (e.g. '3/4') to a percentage string.

    Args:
        fraction_str: String in format 'numerator/denominator'

    Returns:
        Formatted percentage string
    """
    try:
        if "/" not in fraction_str:
            return fraction_str
        num, denom = map(float, fraction_str.split("/"))
        if denom == 0:
            return "0.0%"
        return f"{(num / denom) * 100:>6.1f}%"
    except Exception as e:
        logger.error(f"Error converting fraction to percentage: {str(e)}")
        return fraction_str


def print_benchmark_results(task_name: str, df: pd.DataFrame) -> None:
    """Print benchmark results in a clean, formatted way.

    Args:
        task_name: Name of the benchmark task
        df: DataFrame containing the results
    """
    # Configure pandas display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.precision", 3)

    # Create a pretty header
    header = f" {task_name} Benchmark Results "
    print("\n" + "=" * 80)
    print(header.center(80))
    print("=" * 80 + "\n")

    # Format and print the DataFrame
    # Right-align all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:>8.3f}")

    # Convert fraction strings to percentages
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: fraction_to_percentage(str(x))
            if isinstance(x, str) and "/" in x
            else x
        )

    # Print with consistent spacing
    print(
        df.to_string(
            justify="right",
            col_space=15,
            index_names=False,
            float_format=lambda x: f"{x:>8.3f}",  # Always returns string
        )
    )
    print("\n" + "-" * 80 + "\n")


@app.command()
def generate_results(
    results_data_path: str = "",
    task: str = "",
):
    """Generate and display metrics from benchmark results.

    Args:
        results_data_path: Path to results directory (default: ./results/{task})
        task: Type of task to analyze. If not specified, will show results for all tasks.
    """
    allowed_tasks = [
        "multilabel_classification",
        "ner",
        "synthetic_data_generation",
        "function_calling",
    ]

    if task and task not in allowed_tasks:
        raise ValueError(f"Invalid task: {task}. Allowed values are {allowed_tasks}")

    # If no task specified, run for all tasks
    tasks_to_run = [task] if task else allowed_tasks

    for current_task in tasks_to_run:
        current_results_path = results_data_path or f"./results/{current_task}"

        # Load results
        results = {}
        try:
            if os.path.exists(current_results_path):
                for file_name in os.listdir(current_results_path):
                    if file_name.endswith(".pkl"):
                        file_path = os.path.join(current_results_path, file_name)
                        with open(file_path, "rb") as file:
                            framework_results = pickle.load(file)
                            results.update(framework_results)
            else:
                logger.warning(f"No results found for task: {current_task}")
                continue
        except Exception as e:
            logger.error(
                f"Failed to load results from {current_results_path}: {str(e)}"
            )
            continue

        # Calculate metrics
        try:
            if current_task == "multilabel_classification":
                metrics_data = {}
                for framework, value in results.items():
                    framework_name = framework.replace("Framework", "")

                    # Calculate latency p95
                    flat_latencies = list(itertools.chain(*value["latencies"]))
                    latency_p95 = (
                        np.percentile(flat_latencies, 95) if flat_latencies else 0
                    )

                    # Calculate accuracy if metrics exist
                    accurate_count = 0
                    total_count = 0
                    if value.get("metrics"):
                        if isinstance(value["metrics"], dict):
                            accurate_count = int(
                                value["metrics"].get("accuracy", 0)
                                * len(value["predictions"])
                            )
                            total_count = len(value["predictions"])
                        elif isinstance(value["metrics"], list):
                            # If metrics is a list, take the first item
                            accuracy = (
                                value["metrics"][0].get("accuracy", 0)
                                if value["metrics"]
                                else 0
                            )
                            accurate_count = int(accuracy * len(value["predictions"]))
                            total_count = len(value["predictions"])

                    metrics_data[framework_name] = [
                        latency_p95,
                        f"{sum(value['percent_successful'])}/{len(value['percent_successful'])}",
                        f"{accurate_count}/{total_count}" if total_count > 0 else "0/0",
                    ]

                df = pd.DataFrame(
                    metrics_data,
                    index=[
                        "Latency p95(s)",
                        "Schema Compliance",
                        "Label Accuracy",
                    ],
                )
                print_benchmark_results("Multilabel Classification", df)

            elif current_task == "ner":
                metrics_data = {}
                for framework, value in results.items():
                    framework_name = framework.replace("Framework", "")

                    # Calculate latency p95
                    flat_latencies = list(itertools.chain(*value["latencies"]))
                    latency_p95 = (
                        np.percentile(flat_latencies, 95) if flat_latencies else 0
                    )

                    # Calculate NER metrics
                    tp_total, fp_total, fn_total = 0, 0, 0
                    if value.get("metrics"):
                        for run in value["metrics"]:
                            for metric in run:
                                tp_total += sum(metric["true_positives"].values())
                                fp_total += sum(metric["false_positives"].values())
                                fn_total += sum(metric["false_negatives"].values())

                    metrics_data[framework_name] = [
                        latency_p95,
                        f"{sum(value['percent_successful'])}/{len(value['percent_successful'])}",
                        f"{tp_total}/{tp_total + fp_total}"
                        if (tp_total + fp_total) > 0
                        else "0/0",
                        f"{tp_total}/{tp_total + fn_total}"
                        if (tp_total + fn_total) > 0
                        else "0/0",
                        f"{2 * tp_total}/{2 * tp_total + fp_total + fn_total}"
                        if (tp_total + fp_total + fn_total) > 0
                        else "0/0",
                    ]

                df = pd.DataFrame(
                    metrics_data,
                    index=[
                        "Latency p95(s)",
                        "Schema Compliance",
                        "Precision",
                        "Recall",
                        "F1",
                    ],
                )
                print_benchmark_results("Named Entity Recognition", df)

            elif current_task == "synthetic_data_generation":
                metrics_data = {}
                for framework, value in results.items():
                    framework_name = framework.replace("Framework", "")
                    # Calculate latency p95
                    flat_latencies = list(itertools.chain(*value["latencies"]))
                    latency_p95 = (
                        np.percentile(flat_latencies, 95) if flat_latencies else 0
                    )
                    # Calculate variety
                    variety = 0
                    if value.get("predictions") and value["predictions"][0]:
                        calls = value["predictions"][0]
                        unique_functions = len({call.get("name", "") for call in calls})
                        total_calls = len(calls)
                        variety = (
                            f"{unique_functions}/{total_calls}" if calls else "0/0"
                        )

                    metrics_data[framework_name] = [
                        latency_p95,
                        f"{sum(value['percent_successful'])}/{len(value['percent_successful'])}",
                        variety,
                    ]

                df = pd.DataFrame(
                    metrics_data,
                    index=[
                        "Latency p95(s)",
                        "Schema Compliance",
                        "Output Diversity",
                    ],
                )
                print_benchmark_results("Synthetic Data Generation", df)

            elif current_task == "function_calling":
                metrics_data = {}
                for framework, value in results.items():
                    metrics_list = value.get("metrics", [])
                    if not metrics_list:  # Check if the list is empty
                        continue

                    framework_name = framework.replace("Framework", "")

                    # Calculate latency p95
                    flat_latencies = list(itertools.chain(*value["latencies"]))
                    latency_p95 = (
                        np.percentile(flat_latencies, 95) if flat_latencies else 0
                    )

                    # Initialize aggregated metrics
                    aggregated_metrics = {
                        "total_calls": 0,
                        "exact_match_ratio_sum": 0,
                        "matching_args_sum": 0,
                        "total_args_sum": 0,
                        "extra_args_sum": 0,
                        "matching_values_sum": 0,
                        "total_values_sum": 0,
                        "avg_similarity_sum": 0,
                        "weighted_score_sum": 0,
                    }

                    # Flatten the nested metrics list structure
                    flattened_metrics = []
                    for metrics_group in metrics_list:
                        if isinstance(metrics_group, list):
                            for metrics_subgroup in metrics_group:
                                if isinstance(metrics_subgroup, list):
                                    flattened_metrics.extend(metrics_subgroup)
                                else:
                                    flattened_metrics.append(metrics_subgroup)
                        else:
                            flattened_metrics.append(metrics_group)

                    # Process the flattened metrics
                    for metrics in flattened_metrics:
                        tool_metrics = metrics["tool_accuracy"]
                        value_metrics = metrics["value_accuracy"]

                        aggregated_metrics["total_calls"] += tool_metrics["total_calls"]
                        aggregated_metrics["exact_match_ratio_sum"] += (
                            tool_metrics["exact_match_ratio"]
                            * tool_metrics["total_calls"]
                        )
                        aggregated_metrics["matching_args_sum"] += len(
                            metrics["detailed_metrics"][0]["arguments"]["matching"]
                        )
                        aggregated_metrics["total_args_sum"] += len(
                            metrics["detailed_metrics"][0]["arguments"]["matching"]
                        ) + len(metrics["detailed_metrics"][0]["arguments"]["missing"])
                        aggregated_metrics["extra_args_sum"] += len(
                            metrics["detailed_metrics"][0]["arguments"]["extra"]
                        )
                        aggregated_metrics["matching_values_sum"] += metrics[
                            "detailed_metrics"
                        ][0]["arguments"]["matching_values"]
                        aggregated_metrics["total_values_sum"] += len(
                            metrics["detailed_metrics"][0]["arguments"]["matching"]
                        )
                        aggregated_metrics["avg_similarity_sum"] += value_metrics[
                            "avg_similarity"
                        ]
                        aggregated_metrics["weighted_score_sum"] += metrics[
                            "overall_quality"
                        ]["weighted_score"]

                    # Calculate averages
                    num_runs = len(flattened_metrics)
                    avg_exact_match_ratio = (
                        aggregated_metrics["exact_match_ratio_sum"]
                        / aggregated_metrics["total_calls"]
                        if aggregated_metrics["total_calls"]
                        else 0
                    )
                    avg_matching_args = (
                        aggregated_metrics["matching_args_sum"] / num_runs
                        if num_runs
                        else 0
                    )
                    avg_total_args = (
                        aggregated_metrics["total_args_sum"] / num_runs
                        if num_runs
                        else 0
                    )  # Use num_runs, as each run has args
                    avg_extra_args = (
                        aggregated_metrics["extra_args_sum"] / num_runs
                        if num_runs
                        else 0
                    )
                    avg_matching_values = (
                        aggregated_metrics["matching_values_sum"]
                        / aggregated_metrics["total_values_sum"]
                        if aggregated_metrics["total_values_sum"]
                        else 0
                    )
                    avg_total_values = (
                        aggregated_metrics["total_values_sum"] / num_runs
                        if num_runs
                        else 0
                    )  # Use num_runs
                    avg_similarity = (
                        aggregated_metrics["avg_similarity_sum"] / num_runs
                        if num_runs
                        else 0
                    )
                    avg_weighted_score = (
                        aggregated_metrics["weighted_score_sum"] / num_runs
                        if num_runs
                        else 0
                    )

                    metrics_data[framework_name] = [
                        latency_p95,
                        f"{sum(value['percent_successful'])}/{len(value['percent_successful'])}",
                        f"{int(avg_exact_match_ratio * aggregated_metrics['total_calls'])}/{aggregated_metrics['total_calls']}",
                        f"{int(avg_matching_args)}/{int(avg_total_args)}",
                        f"{int(avg_total_args - avg_extra_args)}/{int(avg_total_args)}",
                        f"{int(avg_matching_args)}/{int(avg_total_args)}",
                        f"{int(avg_matching_values)}/{int(avg_total_values)}"
                        if avg_total_values > 0
                        else "0/0",
                        avg_similarity,
                        avg_weighted_score,
                    ]

                df = pd.DataFrame(
                    metrics_data,
                    index=[
                        "Latency p95(s)",
                        "Schema Compliance",
                        "Function Match",
                        "Param Coverage",
                        "Param Precision",
                        "Param Recall",
                        "Value Match",
                        "Value Similarity",
                        "Overall Quality",
                    ],
                )
                print_benchmark_results("Function Calling", df)

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.exception(e)
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
