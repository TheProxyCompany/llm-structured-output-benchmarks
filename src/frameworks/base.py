from abc import ABC, abstractmethod
from typing import Any
import json
import pandas as pd
import torch
from loguru import logger

from data_sources.data_models import (
    User,
    FunctionCall,
)
from src.experiment import ExperimentResult


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

    def __init__(self, **kwargs) -> None:
        self.task = kwargs.get("task", "")
        self.prompt = kwargs.get("prompt", "")
        self.llm_model = kwargs.get("llm_model", "gpt-4o-mini")
        self.llm_model_family = kwargs.get("llm_model_family", "openai")
        self.device = kwargs.get("device", "cpu")
        self.source_data_pickle_path = kwargs.get("source_data_pickle_path", "")
        self.sample_rows = kwargs.get("sample_rows", 1)
        self.source_data = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Create the response model
        if self.task == "synthetic_data_generation":
            self.response_model = User
        elif self.task == "function_calling":
            self.response_model = FunctionCall

        logger.info(f"Response model is {self.response_model}")

    @abstractmethod
    def run(
        self, n_runs: int, expected_response: Any, *args, **kwargs
    ) -> ExperimentResult:
        pass

    def run_experiment(
        self,
        task: str,
        n_runs: int,
        row: Any | None = None,
        seeds: list[int] | None = None,
    ) -> ExperimentResult:
        """
        Args:
            task: Name of the task being performed
            n_runs: Number of runs per evaluation
            row: Optional row of test data (can be pandas Series or namedtuple)
            seeds: Optional list of seeds to use for each run
        """

        try:
            if task == "function_calling" and row:
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
                result = self.run(
                    n_runs=n_runs,
                    expected_response=expected,
                    inputs=inputs,
                    seeds=seeds,  # Pass seeds to the run method
                )
            else:
                result = self.run(
                    n_runs=n_runs,
                    expected_response=None,
                    seeds=seeds,  # Pass seeds to the run method
                )

        except Exception as e:
            logger.error(f"Error during framework evaluation: {str(e)}")
            logger.exception(e)
            breakpoint()
            raise e

        return result

    def set_source_data(self, num_rows: int | None = None, seed: int | None = None) -> None:
        if self.source_data_pickle_path:
            self.source_data = pd.read_pickle(self.source_data_pickle_path)
            if num_rows:
                # Use fixed random state for reproducible sampling
                self.source_data = self.source_data.sample(
                    num_rows,
                    random_state=seed,
                )
                self.source_data = self.source_data.reset_index(drop=True)
