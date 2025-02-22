from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from data_sources.data_models import (
    multilabel_classification_model,
    synthetic_data_generation_model,
    function_calling_model,
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
        self.llm_model = kwargs.get("llm_model", "gpt-4o-mini")
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

        elif self.task == "synthetic_data_generation":
            self.response_model = synthetic_data_generation_model()

        elif self.task == "function_calling":
            self.response_model = function_calling_model()

        logger.info(f"Response model is {self.response_model}")

    @abstractmethod
    def run(self, n_runs: int, expected_response: Any, *args, **kwargs): ...
