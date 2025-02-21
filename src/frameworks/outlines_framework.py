from typing import Any

from outlines.generate.json import json as outlines_json
from outlines.models.transformers import transformers as outlines_transformers

from outlines.samplers import MultinomialSampler

from frameworks.base import BaseFramework, experiment


class OutlinesFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_length = kwargs.get("max_length", 4096)

        # TODO: Handle openai model
        if self.llm_model_family == "transformers":
            outlines_model = outlines_transformers(
                self.llm_model,
                device=self.device
            )
            outlines_model.model.config.pad_token_id = (
                outlines_model.model.config.eos_token_id[-1]
            )
            if outlines_model.model.generation_config:
                outlines_model.model.generation_config.pad_token_id = (
                    outlines_model.model.config.eos_token_id[-1]
                )
        else:
            raise ValueError(f"Model family: {self.llm_model_family} not supported")

        sampler = MultinomialSampler(
            top_k=10,
            top_p=None,
            temperature=1.0,
        )
        self.outline_generator = outlines_json(
            outlines_model, self.response_model, sampler=sampler
        )

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            response = self.outline_generator(
                self.prompt.format(
                    json_schema=self.response_model.model_json_schema(),
                    **inputs,
                ),
                max_tokens=self.max_length,
            )
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs) # type: ignore
        return predictions, percent_successful, metrics, latencies
