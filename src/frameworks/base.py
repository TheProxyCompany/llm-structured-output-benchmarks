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
        source_data_pickle_path = kwargs.get("source_data_pickle_path", "")
        self.source_data = None

        # Set random seed for reproducibility
        seed = kwargs.get("seed", 11)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        import random

        np.random.seed(seed)
        random.seed(seed)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Load the data
        if source_data_pickle_path:
            self.source_data = pd.read_pickle(source_data_pickle_path)
            sample_rows = kwargs.get("sample_rows", 0)
            if sample_rows:
                # Use fixed random state for reproducible sampling
                self.source_data = self.source_data.sample(
                    sample_rows,
                    random_state=seed,
                )
                self.source_data = self.source_data.reset_index(drop=True)
            logger.info(f"Loaded source data from {source_data_pickle_path}")

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
        self, task: str, n_runs: int, row: Any | None = None
    ) -> ExperimentResult:
        """
        Args:
            task: Name of the task being performed
            n_runs: Number of runs per evaluation
            row: Optional row of test data (can be pandas Series or namedtuple)
        """

        try:
            if not row:
                result = self.run(n_runs=n_runs, expected_response=None)
                return result

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
                result = self.run(
                    n_runs=n_runs,
                    expected_response=expected,
                    inputs=inputs,
                )
        except Exception as e:
            logger.error(f"Error during framework evaluation: {str(e)}")
            logger.exception(e)

        return result or ExperimentResult([], [], None, n_runs)
