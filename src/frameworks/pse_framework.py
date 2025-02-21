from typing import Any

import json

import torch
from frameworks.base import BaseFramework, experiment
from transformers import LlamaForCausalLM, AutoTokenizer
from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin

class LLamaCausalLMWithPSE(LlamaForCausalLM, PSETorchMixin):
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

        # Add the PSE engine to model
        self.model.engine = StructuringEngine(self.tokenizer)

        self.model.config.pad_token_id = self.model.config.eos_token_id[-1]
        if self.model.generation_config:
            self.model.generation_config.pad_token_id = self.model.config.eos_token_id[-1]

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:

        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs) -> tuple[list[Any], float, dict, list[list[float]]]:
            # Configure engine with schema
            self.model.engine.configure(self.response_model.schema())

            # Format prompt and generate response
            prompt = self.prompt.format(
                json_schema=self.response_model.model_json_schema(),
                **inputs
            )
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
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
            response = self.response_model(**json.loads(response_text.strip()))
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs) # type: ignore
        return predictions, percent_successful, metrics, latencies
