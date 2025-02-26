import json
from typing import Any

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import pipeline

from src.frameworks.base import BaseFramework
from src.experiment import experiment, ExperimentResult


class LMFormatEnforcerFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parser = JsonSchemaParser(self.response_model.schema())
        max_length = kwargs.get("max_length", 4096)
        if self.llm_model_family == "transformers":
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                device_map=self.device,
                max_length=max_length,
                top_p=None,
                do_sample=True,
                temperature=0.0,
                truncation=True,
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

    def run(self, n_runs: int, expected_response: Any = None, inputs: dict = {}) -> ExperimentResult:

        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            prompt = inputs.get("prompt")
            if not prompt:
                prompt = self.prompt.format(
                    json_schema=self.response_model.model_json_schema(), **inputs
                )

            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

            assert self.hf_pipeline.tokenizer is not None
            formatted_prompt = (
                self.hf_pipeline.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            )

            assert isinstance(formatted_prompt, str)
            if "schema" in inputs:
                schema = inputs.get("schema")
                self.parser = JsonSchemaParser(schema)
                self.prefix_function = build_transformers_prefix_allowed_tokens_fn(
                    self.hf_pipeline.tokenizer,  # type: ignore
                    self.parser,
                )

            response = self.hf_pipeline(
                formatted_prompt,
                prefix_allowed_tokens_fn=self.prefix_function
            )
            response = response[0]["generated_text"][len(formatted_prompt) :].strip()  # type: ignore
            response = self.response_model(**json.loads(response))
            return response

        return run_experiment(inputs) # type: ignore
