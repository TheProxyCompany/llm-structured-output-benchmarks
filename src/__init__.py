import os
import pickle

from loguru import logger
import numpy as np
import pandas as pd

from src.experiment import ExperimentResult


def print_benchmark_results(
    task_name: str, num_samples: int, num_runs: int, df: pd.DataFrame
) -> None:
    """Print benchmark results in a clean, formatted way.

    Args:
        task_name: Name of the benchmark task
        num_samples: Number of samples in the benchmark
        num_runs: Number of runs in the benchmark
        df: DataFrame containing the results
    """
    # Configure pandas display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.precision", 3)
    # Create a pretty header
    if num_samples == -1:
        header = (
            f" {task_name} Benchmark Results ({num_runs} Samples) "
        )
    else:
        header = (
            f" {task_name} Benchmark Results ({num_samples} Samples x {num_runs} Runs) "
        )
    print("\n" + "=" * 80)
    print(header.center(80))
    print("=" * 80 + "\n")

    # Format and print the DataFrame
    # Right-align all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:>8.3f}")

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


def save_results(results: list[ExperimentResult], task: str, framework_name: str):
    """Save benchmark results to disk.

    Args:
        results: List of ExperimentResult objects
        task: Name of the task (determines subdirectory)
        framework_name: Name of the framework (determines filename)
    """
    directory = f"results/{task}"
    os.makedirs(directory, exist_ok=True)

    results_df = pd.DataFrame([result.to_dict() for result in results]).T
    output_path = f"{directory}/{framework_name}.pkl"
    with open(output_path, "wb") as file:
        pickle.dump(results_df, file)
        logger.info(f"Results saved to {output_path}")
