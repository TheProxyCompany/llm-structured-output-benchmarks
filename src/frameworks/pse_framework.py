import logging
from typing import Any

import json

import torch
from src.frameworks.base import BaseFramework
from src.experiment import experiment, ExperimentResult

from transformers import LlamaForCausalLM, AutoTokenizer
from pse.engine.structuring_engine import StructuringEngine
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

        if self.llm_model_family != "transformers":
            raise ValueError(f"Model family: {self.llm_model_family} not supported")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        self.model = LLamaCausalLMWithPSE.from_pretrained(
            self.llm_model,
            device_map=self.device,
        )

        self.model.config.pad_token_id = self.model.config.eos_token_id[-1]
        if self.model.generation_config:
            self.model.generation_config.pad_token_id = self.model.config.eos_token_id[-1]

        self.model.engine = StructuringEngine(self.tokenizer, multi_token_sampling=True)
        if not self.task == "function_calling":
            self.model.engine.configure(self.response_model.schema())

    def run(self, n_runs: int, expected_response: Any = None, inputs: dict = {}) -> ExperimentResult:

        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs) -> tuple[list[Any], float, dict, list[list[float]]]:
            prompt = inputs.get("prompt")
            if not prompt:
                prompt = self.prompt.format(
                    json_schema=self.response_model.model_json_schema(),
                    **inputs
                )
            # Configure engine with schema
            if "schema" in inputs:
                schema = inputs.get("schema")
                self.model.engine.configure(schema)
            else:
                self.model.engine.reset()

            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

            input_ids = self.tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
            # 5. Generate!
            assert isinstance(input_ids, torch.Tensor)
            input_ids = input_ids.to(self.model.device)
            assert isinstance(input_ids, torch.Tensor)
            # Generate with PSE constraints
            output_ids = self.model.generate(
                input_ids,
                max_length=self.max_length,
                top_p=None,
                top_k=10,
                do_sample=True,
                temperature=1.0,
            )
            # Decode and parse response
            response_text = self.tokenizer.decode(output_ids[0][len(input_ids[0]) :])
            try:
                response = self.response_model(**json.loads(response_text))
            except Exception as e:
                print(f"Error parsing response: {e}")
                print(f"Response text: {response_text}")
                raise e
            return response

        return run_experiment(inputs) # type: ignore
