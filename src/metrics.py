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

def compare_framework_results(results: dict[str, pd.DataFrame], task: str) -> tuple[pd.DataFrame, int, int]:
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
                framework_metrics = calculate_synthetic_data_generation_metrics(framework_results)
            else:
                raise ValueError(f"Metrics not implemented for task: {task}")
            comparison[framework_name] = framework_metrics.metrics
            num_samples = framework_metrics.num_samples
            num_runs = framework_metrics.num_runs
        except Exception as e:
            logger.error(f"Error calculating metrics for {framework_name}: {e}")
            continue

    # Create comparison DataFrame with frameworks as columns
    comparison_df = pd.DataFrame.from_dict(comparison)

    # Add metric descriptions as index names
    metric_descriptions = {
        "completion_rate": "Valid Schema Rate",
        "p50_latency": "Average Latency (s)",
        "p95_latency": "95th Percentile Latency (s)",
        "average_name_match": "Average Name Match",
        "average_args_match": "Average Args Match",
        "average_args_correctness": "Average Args Correctness",
    }
    # Use a list comprehension with a default value for robustness.
    index_names = [
        metric_descriptions.get(i, i) for i in comparison_df.index
    ]
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
    average_args_correctness = []

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
            np.mean([name_match for name_match, _, _ in numerical_scores])
        )
        average_args_matches.append(
            np.mean([args_match for _, args_match, _ in numerical_scores])
        )
        average_args_correctness.append(
            np.mean([args_correctness for _, _, args_correctness in numerical_scores])
        )
    # Latency metrics
    metrics["p50_latency"] = np.percentile(average_latencies, 50)
    metrics["p95_latency"] = np.percentile(average_latencies, 95)
    # Consistency metrics
    metrics["completion_rate"] = np.mean(completion_rates) * 100
    metrics["average_name_match"] = np.mean(average_name_matches) * 100
    metrics["average_args_match"] = np.mean(average_args_matches) * 100
    metrics["average_args_correctness"] = np.mean(average_args_correctness) * 100

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

    metrics["completion_rate"] = np.mean(completion_rates) * 100
    metrics["average_latencies"] = np.mean(average_latencies)
    metrics["sample_scores"] = np.mean(sample_scores) * 100

    return FrameworkMetrics(metrics, -1, num_runs)
