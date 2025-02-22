import os
import pickle

import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from tqdm import tqdm

from src.frameworks import factory
from src import save_results, print_benchmark_results

app = typer.Typer()


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
        for config in config_values:
            results = []
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
                for row in tqdm(
                    framework_instance.source_data.itertuples(),
                    desc=f"Running {framework_instance.task}",
                    total=len(framework_instance.source_data),
                ):
                    experiment_result = framework_instance.run_experiment(
                        task=task,
                        n_runs=n_runs,
                        row=row,
                    )
                    results.append(experiment_result)
            else:
                experiment_result = framework_instance.run_experiment(
                    task=task,
                    n_runs=n_runs,
                )
                results.append(experiment_result)

            # Save results after each framework evaluation
            save_results(results, task, config_key)


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

        results = pd.DataFrame(results)
        # Calculate metrics
        breakpoint()
        try:
            if current_task == "multilabel_classification":
                print_benchmark_results(
                    "Multilabel Classification",
                    num_samples=len(
                        results["Multilabel Classification"]["predictions"]
                    ),
                    num_runs=len(results["Multilabel Classification"]["latencies"]),
                    df=results,
                )

            elif current_task == "synthetic_data_generation":
                print_benchmark_results(
                    "Synthetic Data Generation",
                    num_samples=len(
                        results["Synthetic Data Generation"]["predictions"]
                    ),
                    num_runs=len(results["Synthetic Data Generation"]["latencies"]),
                    df=results,
                )

            elif current_task == "function_calling":
                print_benchmark_results(
                    "Function Calling",
                    num_samples=len(results["Function Calling"]["predictions"]),
                    num_runs=len(results["Function Calling"]["latencies"]),
                    df=results,
                )

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.exception(e)
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
