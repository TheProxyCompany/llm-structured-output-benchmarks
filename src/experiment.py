
import logging
import time
from typing import Any, Callable

import torch
from tqdm import tqdm

from src.metrics import calculate_function_calling_metrics
from src.frameworks.base import response_parsing

logger = logging.getLogger(__name__)

def experiment(
    n_runs: int = 10,
    expected_response: Any = None,
    task: str = "multilabel_classification",
) -> Callable[
    ...,
    tuple[list[Any], float, dict[str, Any] | list[dict[str, Any]] | None, list[float]],
]:
    """Decorator to run an LLM call function multiple times and return the responses

    Args:
        n_runs (int): Number of times to run the function
        expected_response (Any): The expected response. If provided, the decorator will calculate accurary too.
        task (str): The task being performed. Default is "multilabel_classification".
                   Available options are "multilabel_classification", "ner", "synthetic_data_generation", and "function_calling"

    Returns:
        Callable[..., tuple[list[Any], float, dict[str, Any] | list[dict[str, Any]] | None, list[float]]]:
        A function that returns:
        - List of outputs from the function runs
        - Percent of successful runs (float between 0 and 1)
        - Metrics if expected_response is provided, else None
        - List of latencies for each successful call
    """

    def experiment_decorator(func):
        def wrapper(*args, **kwargs):
            allowed_tasks = [
                "multilabel_classification",
                "synthetic_data_generation",
                "function_calling",
            ]
            if task not in allowed_tasks:
                raise ValueError(
                    f"{task} is not allowed. Allowed values are {allowed_tasks}"
                )

            responses, latencies = [], []
            for _ in tqdm(range(n_runs), leave=False):
                try:
                    torch.mps.empty_cache() 
                    start_time = time.time()
                    response = func(*args, **kwargs)
                    end_time = time.time()

                    response = response_parsing(response)

                    if "classes" in response:
                        response = response_parsing(response["classes"])

                    responses.append(response)
                    latencies.append(end_time - start_time)
                except Exception as e:
                    import traceback

                    logger.error(f"Error in experiment: {e}")
                    logger.error(traceback.format_exc())
                    pass

            num_successful = len(responses)
            percent_successful = num_successful / n_runs

            # Metrics calculation
            if task == "multilabel_classification" and expected_response:
                accurate = 0
                for response in responses:
                    if response == expected_response:
                        accurate += 1

                framework_metrics = {
                    "accuracy": accurate / num_successful if num_successful else 0
                }

            elif task == "function_calling":
                framework_metrics = []
                for response in responses:
                    # Don't wrap in list since calculate_function_calling_metrics handles single calls
                    metrics = calculate_function_calling_metrics(
                        expected_response, response
                    )
                    framework_metrics.append(
                        [metrics]
                    )  # Keep outer list for consistency with data structure

            return (
                responses,
                percent_successful,
                framework_metrics if expected_response else None,
                latencies,
            )

        return wrapper

    return experiment_decorator  # type: ignore
