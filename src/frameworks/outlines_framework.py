from typing import Any
import json

from outlines.generate.json import json as outlines_json
from outlines.models.transformers import transformers as outlines_transformers

from outlines.samplers import GreedySampler, MultinomialSampler

from src.frameworks.base import BaseFramework
from src.experiment import experiment, ExperimentResult


class OutlinesFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_length = kwargs.get("max_length", 4096)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", None)

        self.outlines_model = outlines_transformers(
            self.llm_model,
            device=self.device
        )
        self.outlines_model.model.config.pad_token_id = (
            self.outlines_model.model.config.eos_token_id[-1]
        )
        if self.outlines_model.model.generation_config:
            self.outlines_model.model.generation_config.do_sample = self.do_sample
            self.outlines_model.model.generation_config.temperature = self.temperature
            self.outlines_model.model.generation_config.pad_token_id = (
                self.outlines_model.model.config.eos_token_id[-1]
            )

        if self.do_sample or self.temperature:
            self.sampler = MultinomialSampler(temperature=self.temperature)
        else:
            self.sampler = GreedySampler()

        self.outline_generator = outlines_json(
            self.outlines_model, self.response_model, sampler=self.sampler
        )

    def run(
        self,
        n_runs: int,
        expected_response: Any = None,
        inputs: dict = {},
        seeds: list[int] | None = None,
    ) -> ExperimentResult:

        @experiment(n_runs=n_runs, expected_response=expected_response, seeds=seeds)
        def run_experiment(inputs):
            prompt = inputs.get("prompt")
            if not prompt:
                prompt = self.prompt.format(
                    json_schema=self.response_model.model_json_schema(), **inputs
                )

            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

            formatted_prompt = (
                self.outlines_model.tokenizer.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            )
            assert isinstance(formatted_prompt, str)
            if "schema" in inputs:
                schema = json.dumps(inputs.get("schema"))
                self.outline_generator = outlines_json(self.outlines_model, schema, sampler=self.sampler)

            response = self.outline_generator(
                formatted_prompt,
                max_tokens=self.max_length,
            )
            return response

        return run_experiment(inputs) # type: ignore
