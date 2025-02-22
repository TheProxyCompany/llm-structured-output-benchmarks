# 🧩 LLM Structured Output Benchmarks

Benchmark various *local* Structured Output frameworks:
    - Outlines
    - LMFormatEnforcer
    - Proxy Structuring Engine

This is a quick fork from the original [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) repository.


## 🏆 Benchmark Results [2025-02-22]

<sup>*</sup> Macbook Pro M3 128GB RAM

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

   # Multilabel classification
   python -m main generate-results --task multilabel_classification
   ```

## Benchmark methods

1. Synthetic Data Generation
    - **Task**: Generate synthetic data similar according to a Pydantic data model schema.
    - **Data**:
        - Two level nested User details Pydantic schema.
    - **Prompt**: `Generate a random person's information. The name must be chosen at random. Make it something you wouldn't normally choose.`
    - **Evaluation Metrics**:
        1. Reliability: The percentage of times the framework returns valid labels without errors. The average of all the rows `percent_successful` values.
        1. Latency: The 95th percentile of the time taken to run the framework on the data.
        1. Variety: The percent of names that are unique compared to all names generated.
    - **Experiment Details**: Run each row through the framework `n_runs` number of times and log the percent of successful runs.
