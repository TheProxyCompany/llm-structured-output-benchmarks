import time
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from data_sources.data_models import (
    multilabel_classification_model,
    ner_model,
    synthetic_data_generation_model,
    function_calling_model,
)
from src.frameworks.metrics import (
    calculate_metrics,
    calculate_function_calling_metrics,
)


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
                "ner",
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
            elif task == "ner":
                framework_metrics = []
                for response in responses:
                    framework_metrics.append(
                        calculate_metrics(expected_response, response)
                    )

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


class BaseFramework(ABC):
    task: str
    prompt: str
    llm_model: str
    llm_model_family: str
    retries: int
    source_data_pickle_path: str
    sample_rows: int
    response_model: Any
    device: str

    def __init__(self, *args, **kwargs) -> None:
        self.task = kwargs.get("task", "")
        self.prompt = kwargs.get("prompt", "")
        self.llm_model = kwargs.get("llm_model", "gpt-3.5-turbo")
        self.llm_model_family = kwargs.get("llm_model_family", "openai")
        self.retries = kwargs.get("retries", 0)
        self.device = kwargs.get("device", "cpu")
        source_data_pickle_path = kwargs.get("source_data_pickle_path", "")

        # Load the data
        if source_data_pickle_path:
            self.source_data = pd.read_pickle(source_data_pickle_path)

            sample_rows = kwargs.get("sample_rows", 0)
            if sample_rows:
                self.source_data = self.source_data.sample(sample_rows)
                self.source_data = self.source_data.reset_index(drop=True)
            logger.info(f"Loaded source data from {source_data_pickle_path}")
        else:
            self.source_data = None

        # Create the response model
        if "response_model" in kwargs:
            self.response_model = kwargs["response_model"]
        elif self.task == "multilabel_classification":
            # Identify the classes
            assert self.source_data is not None
            if isinstance(self.source_data.iloc[0]["labels"], list):
                self.classes = self.source_data["labels"].explode().unique()
            else:
                self.classes = self.source_data["labels"].unique()
            logger.info(
                f"Source data has {len(self.source_data)} rows and {len(self.classes)} classes"
            )

            self.response_model = multilabel_classification_model(self.classes)

        elif self.task == "ner":
            # Identify the entities
            assert self.source_data is not None
            self.entities = list(
                {key for d in self.source_data["labels"] for key in d.keys()}
            )

            self.response_model = ner_model(self.entities)

        elif self.task == "synthetic_data_generation":
            self.response_model = synthetic_data_generation_model()

        elif self.task == "function_calling":
            self.response_model = function_calling_model()

        logger.info(f"Response model is {self.response_model}")

    @abstractmethod
    def run(self, n_runs: int, expected_response: Any, *args, **kwargs): ...
