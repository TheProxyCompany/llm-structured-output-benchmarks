import logging

# import sys
from typing import Any

import json

import torch
from src.frameworks.base import BaseFramework
from src.experiment import experiment, ExperimentResult

from transformers import LlamaForCausalLM, AutoTokenizer
from pse.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.DEBUG,
#     stream=sys.stdout
# )


class LLamaCausalLMWithPSE(LlamaForCausalLM, PSETorchMixin):
    """
    This is an easy way to integrate the StructuringEngine's functionality
    into a LlamaForCausalLM model.
    """

    pass


class PSEFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_length = kwargs.get("max_length", 4096)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", None)
        if self.llm_model_family != "transformers":
            raise ValueError(f"Model family: {self.llm_model_family} not supported")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        self.model = LLamaCausalLMWithPSE.from_pretrained(
            self.llm_model,
            device_map=self.device,
        )

        self.model.config.pad_token_id = self.model.config.eos_token_id[-1]
        if self.model.generation_config:
            self.model.generation_config.pad_token_id = self.model.config.eos_token_id[
                -1
            ]
            self.model.generation_config.do_sample = self.do_sample
            self.model.generation_config.temperature = self.temperature

        self.model.engine = StructuringEngine(
            self.tokenizer,
            multi_token_sampling=True,
            # max_resample_attempts=3,
        )
        if not self.task == "function_calling":
            self.model.engine.configure(self.response_model)

    def run(
        self,
        n_runs: int,
        expected_response: Any = None,
        inputs: dict = {},
        seeds: list[int] | None = None,
    ) -> ExperimentResult:
        @experiment(n_runs=n_runs, expected_response=expected_response, seeds=seeds)
        def run_experiment(inputs: dict) -> tuple[list[Any], float, dict, list[list[float]]]:
            prompt = inputs.get("prompt")
            if not prompt:
                prompt = self.prompt.format(
                    json_schema=self.response_model.model_json_schema(), **inputs
                )
            # Configure engine with schema
            if "schema" in inputs:
                schema = json.dumps(inputs.get("schema"))
                self.model.engine.configure(schema)
            else:
                self.model.engine.reset()

            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

            input_ids = self.tokenizer.apply_chat_template(
                prompt, return_tensors="pt", add_generation_prompt=True
            )
            # 5. Generate!
            assert isinstance(input_ids, torch.Tensor)
            input_ids = input_ids.to(self.model.device)
            assert isinstance(input_ids, torch.Tensor)
            # Generate with PSE constraints
            _ = self.model.generate(
                input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=None,
                do_sample=self.do_sample,
            )

            return self.model.engine.get_structured_output()

        return run_experiment(inputs)  # type: ignore
