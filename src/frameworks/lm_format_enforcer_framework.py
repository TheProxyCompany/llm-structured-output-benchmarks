import json
from typing import Any

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
import torch
from transformers import pipeline

from frameworks.base import BaseFramework, experiment


class LMFormatEnforcerFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parser = JsonSchemaParser(self.response_model.schema())
        max_length = kwargs.get("max_length", 4096)
        torch.mps.empty_cache()
        if self.llm_model_family == "transformers":
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                device_map=self.device,
                max_length=max_length,
                top_p=None,
                top_k=10,
                do_sample=True,
                temperature=1.0,
            )
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(
                self.hf_pipeline.tokenizer,  # type: ignore
                self.parser
            )
            self.hf_pipeline.model.config.pad_token_id = (
                self.hf_pipeline.model.config.eos_token_id[-1]
            )
            if self.hf_pipeline.model.generation_config:
                self.hf_pipeline.model.generation_config.pad_token_id = (
                    self.hf_pipeline.model.config.eos_token_id[-1]
                )
        else:
            raise ValueError(f"Model family: {self.llm_model_family} not supported")

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            prompt = self.prompt.format(
                json_schema=self.response_model.schema(), **inputs
            )
            response = self.hf_pipeline(
                prompt, prefix_allowed_tokens_fn=self.prefix_function
            )
            response = response[0]["generated_text"][len(prompt) :].strip()  # type: ignore
            response = self.response_model(**json.loads(response))
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)  # type: ignore
        return predictions, percent_successful, metrics, latencies
