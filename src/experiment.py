from dataclasses import asdict, dataclass, is_dataclass
import logging
import time
from enum import Enum
from typing import Any, Callable


from pydantic import BaseModel
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    responses: list[Any]
    latencies: list[float]
    expected_response: Any
    n_runs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_response": self.expected_response,
            "responses": self.responses,
            "latencies": self.latencies,
            "n_runs": self.n_runs,
        }

def experiment(
    n_runs: int = 10,
    expected_response: Any = None,
) -> Callable[
    ...,
    ExperimentResult,
]:
    """Decorator to run an LLM call function multiple times and return the responses

    Args:
        n_runs (int): Number of times to run the function
        expected_response (Any): The expected response. If provided, the decorator will calculate accurary too.
        task (str): The task being performed. Default is "multilabel_classification".
                   Available options are "multilabel_classification", "synthetic_data_generation", and "function_calling"

    Returns:
        Callable[..., ExperimentResult]:
        A function that returns:
        - List of outputs from the function runs
        - List of latencies for each successful call
    """

    def experiment_decorator(
        func: Callable[..., Any],
    ) -> Callable[..., ExperimentResult]:
        def wrapper(*args, **kwargs) -> ExperimentResult:
            responses, latencies = [], []
            for _ in tqdm(range(n_runs), leave=False):
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    start_time = time.time()
                    response = func(*args, **kwargs)
                    end_time = time.time()

                    response = response_parsing(response)

                    responses.append(response)
                    latencies.append(end_time - start_time)
                except Exception as e:
                    import traceback

                    logger.error(f"Error in experiment: {e}")
                    logger.error(traceback.format_exc())

            return ExperimentResult(
                responses=responses,
                latencies=latencies,
                expected_response=expected_response,
                n_runs=n_runs,
            )

        return wrapper

    return experiment_decorator  # type: ignore


def response_parsing(response: Any) -> Any:
    """Parse response into a consistent format.

    Args:
        response: Raw response from model

    Returns:
        Parsed response in dictionary format
    """
    if isinstance(response, list):
        response = {
            member.value if isinstance(member, Enum) else member for member in response
        }
    elif is_dataclass(response):
        response = asdict(response)  # type: ignore
    elif isinstance(response, BaseModel):
        response = response.model_dump(exclude_none=True)
    return response
