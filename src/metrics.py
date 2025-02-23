from dataclasses import dataclass
from typing import cast
import numpy as np
import pandas as pd
from loguru import logger

from data_sources.data_models import FunctionCall, User


@dataclass
class FrameworkMetrics:
    metrics: dict[str, float]
    num_samples: int
    num_runs: int


def compare_framework_results(
    results: dict[str, pd.DataFrame], task: str
) -> tuple[pd.DataFrame, int, int]:
    """Compare metrics across different frameworks and return complete analysis package.

    Args:
        results: Dictionary mapping framework names to their results DataFrames

    Returns:
        Tuple containing:
        - DataFrame with frameworks as columns and metrics as rows
        - Number of samples in the benchmark
        - Number of runs per sample
    """
    comparison = {}
    num_samples: int = 0
    num_runs: int = 0

    for framework_name, framework_results in results.items():
        try:
            if task == "function_calling":
                framework_metrics = calculate_function_call_metrics(framework_results)
            elif task == "synthetic_data_generation":
                framework_metrics = calculate_synthetic_data_generation_metrics(
                    framework_results
                )
            else:
                raise ValueError(f"Metrics not implemented for task: {task}")
            comparison[framework_name] = framework_metrics.metrics
            num_samples = framework_metrics.num_samples
            num_runs = framework_metrics.num_runs
        except Exception as e:
            logger.error(f"Error calculating metrics for {framework_name}: {e}")
            continue

    # Create comparison DataFrame with frameworks as columns, specifying object dtype
    comparison_df = pd.DataFrame.from_dict(comparison, dtype=object)

    # Define value formatting rules
    value_formatters = {
        # Percentage metrics (0.95 -> 95.0%)
        "error_rate": lambda x: f"{x * 100:.1f}%",
        "average_name_match": lambda x: f"{x * 100:.1f}%",
        "average_args_match": lambda x: f"{x * 100:.1f}%",
        "diversity_score": lambda x: f"{x * 100:.1f}%",
        # Latency metrics in seconds with 3 decimal places
        "p50_latency": lambda x: f"{x:.3f}s",
        "p95_latency": lambda x: f"{x:.3f}s",
    }

    # Apply formatting to each metric row
    for metric_key in comparison_df.index:
        if metric_key in value_formatters:
            comparison_df.loc[metric_key] = comparison_df.loc[metric_key].apply(
                value_formatters[metric_key]
            )

    # Update metric descriptions to remove redundant units
    metric_descriptions = {
        "error_rate": "Error Rate",
        "p50_latency": "Average Latency",
        "p95_latency": "95th Percentile Latency",
        "average_name_match": "Average Name Match",
        "average_args_match": "Average Args Match",
        "diversity_score": "Diversity Score",
    }

    # Create descriptive index names
    index_names = [metric_descriptions.get(i, i) for i in comparison_df.index]
    comparison_df.index = pd.Index(index_names, name="Metric")

    return comparison_df, num_samples, num_runs


def calculate_function_call_metrics(
    df: pd.DataFrame,
) -> FrameworkMetrics:
    """Calculate sophisticated metrics for a single framework's results.

    Focuses on real-world applicable metrics rather than simple accuracy.
    """
    num_samples: int = len(df.columns)
    num_runs: int = cast(int, df.loc["n_runs", 0])

    metrics = {}

    # Process responses and latencies per sample
    average_latencies = []
    completion_rates = []
    average_name_matches = []
    average_args_matches = []

    for sample_idx in range(num_samples):
        # Safely extract responses and latencies
        expected_responses_val = df.loc["expected_response", sample_idx]
        expected_function_call = FunctionCall.model_validate(expected_responses_val)
        responses_val = df.loc["responses", sample_idx]
        latencies_val = df.loc["latencies", sample_idx]
        assert isinstance(expected_responses_val, dict)
        assert isinstance(responses_val, list)
        assert isinstance(latencies_val, list)

        latencies = [float(latency) for latency in latencies_val]
        completion_rate = len(responses_val) / num_runs
        completion_rates.append(completion_rate)
        average_latencies.append(np.mean(latencies))

        numerical_scores = [
            FunctionCall.compare(
                expected_function_call,
                FunctionCall.model_validate(generated_response),
            )
            for generated_response in responses_val
        ]
        average_name_matches.append(
            np.mean([name_match for name_match, _ in numerical_scores])
        )
        average_args_matches.append(
            np.mean([args_match for _, args_match in numerical_scores])
        )
    # Latency metrics
    metrics["p50_latency"] = np.percentile(average_latencies, 50)
    metrics["p95_latency"] = np.percentile(average_latencies, 95)
    # Consistency metrics
    metrics["error_rate"] = 1 - np.mean(completion_rates)
    metrics["average_name_match"] = np.mean(average_name_matches)
    metrics["average_args_match"] = np.mean(average_args_matches)

    return FrameworkMetrics(metrics, num_samples, num_runs)


def calculate_synthetic_data_generation_metrics(df: pd.DataFrame) -> FrameworkMetrics:
    """Calculate metrics for synthetic data generation."""
    num_samples: int = len(df.columns)
    num_runs: int = cast(int, df.loc["n_runs", 0])

    metrics = {}
    completion_rates = []
    average_latencies = []
    sample_scores = []
    for sample_idx in range(num_samples):
        # Safely extract responses and latencies
        responses_val = df.loc["responses", sample_idx]
        latencies_val = df.loc["latencies", sample_idx]
        assert isinstance(responses_val, list)
        assert isinstance(latencies_val, list)

        latencies = [float(latency) for latency in latencies_val]
        completion_rate = len(responses_val) / num_runs
        completion_rates.append(completion_rate)
        average_latencies.append(np.mean(latencies))

        users = [User.model_validate(response) for response in responses_val]

        sample_scores.append(User.calculate_diversity_score(users))

    metrics["p50_latency"] = np.percentile(average_latencies, 50)
    metrics["p95_latency"] = np.percentile(average_latencies, 95)
    metrics["error_rate"] = 1 - np.mean(completion_rates)
    metrics["diversity_score"] = np.mean(sample_scores)

    return FrameworkMetrics(metrics, -1, num_runs)
