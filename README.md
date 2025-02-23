# 🧩 LLM Structured Output Benchmarks

Benchmark various *local* Structured Output frameworks:
    - Outlines
    - LMFormatEnforcer
    - Proxy Structuring Engine

This is a quick fork from the original [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) repository.

## Run the benchmarks

1. Install the requirements:
   ```bash
   uv pip install -e .
   ```

2. Run the benchmark:
   ```bash
   python -m main run-benchmark
   ```

3. Raw results will be saved to the `results/` directory

4. Generate results:

   For all tasks:
   ```bash
   python -m main generate-results
   ```

   For specific tasks:
   ```bash
   # Function calling
   python -m main generate-results --task function_calling

   # Synthetic data generation
   python -m main generate-results --task synthetic_data_generation
   ```

## Benchmark methods

- **Model**: meta-llama/Llama-3.1-8b-Instruct

1. Synthetic Data Generation
    - **Task**: Generate synthetic data similar according to a Pydantic data model schema.
    - **Data**:
        - Two level nested User details Pydantic schema.
    - **Prompt**: `Generate a random person's information. The name must be chosen at random. Make it something you wouldn't normally choose.`
    - **Experiment Details**:
        - Run each sample through the framework `n_runs` times
        - Results saved as pickled dataframes for analysis

2. Function Calling
    - **Task**: Evaluate the framework's ability to parse natural language into structured function calls
    - **Data**:
        - Sourced from fireworks-ai/function-calling-eval-dataset-v0
        - Contains prompts, expected function call completions, and available tools/functions
    - **Prompt**: The prompt is a user message that includes the available tools & definitions, and the user's query.
    - **Experiment Details**:
        - Run each sample through the framework `n_runs` times
        - Results saved as pickled dataframes for analysis
