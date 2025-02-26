# 🧩 LLM Structured Output Benchmarks

Benchmark various Structured Output frameworks:
    - Outlines
    - LMFormatEnforcer
    - Proxy Structuring Engine

This is a quick fork from the original [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) repository.

## 🏆 Benchmark Results [2025-02-26]

### Function Calling Results

| Metric | PSEFramework | LMFormatEnforcerFramework | OutlinesFramework |
|:------|:------------:|:-------------------------:|:-----------------:|
| Error Rate | **0.0%** | 0.4% | **0.0%** |
| Average Generation Time | **7.150s** | 10.328s | 16.298s |
| 95th Percentile Generation Time | **8.768s** | 12.119s | 17.975s |
| Average Correct Function Name | **82.3%** | 80.8% | 80.1% |
| Average Correct Function Args | **78.0%** | 76.7% | 77.0% |

### Synthetic Data Generation Results

| Metric | PSEFramework | LMFormatEnforcerFramework | OutlinesFramework |
|:------|:------------:|:-------------------------:|:-----------------:|
| Error Rate | **0.0%** | 1.0% | **0.0%** |
| Average Generation Time | 5.100s | 4.310s | **3.346s** |
| 95th Percentile Generation Time | 5.784s | 4.753s | **4.084s** |
| Diversity Score | **89.0%** | 87.9% | 88.0% |

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
    - **Evaluation Metrics**:
        - Error Rate: Percentage of failed completions (1 - completion rate)
        - Average Generation Time: Median (P50) latency across all generations in seconds
        - 95th Percentile Generation Time: P95 latency indicating worst-case performance
        - Diversity Score: Composite metric (0-100%) measuring variety in generated data, calculated from:
            - Name uniqueness ratio
            - Geographic diversity (unique city/country combinations)

2. Function Calling
    - **Task**: Evaluate the framework's ability to parse natural language into structured function calls
    - **Data**:
        - Sourced from fireworks-ai/function-calling-eval-dataset-v0
        - Contains prompts, expected function call completions, and available tools/functions
    - **Prompt**: The prompt is a user message that includes the available tools & definitions, and the user's query.
    - **Experiment Details**:
        - Run each sample through the framework `n_runs` times
        - Results saved as pickled dataframes for analysis
    - **Evaluation Metrics**:
        - Error Rate: Percentage of failed completions (1 - completion rate)
        - Average Generation Time: Median (P50) latency across all generations in seconds
        - 95th Percentile Generation Time: P95 latency indicating worst-case performance
        - Average Correct Function Name: Percentage of exact matches with expected function names
        - Average Correct Function Args: Score (0-100%) based on expected argument presence and matching


All benchmarks were conducted under identical conditions using an Apple M2 Ultra machine with 192GB RAM.
We used PyTorch and the MPS backend for all benchmarks, using `meta-llama/Llama-3.1-8b-Instruct` as the evaluation LLM.
We intend for these results to be reproducible by anyone who follows the instructions in the README.

## 🎯 Scope

This repository specifically focuses on benchmarking **local** structured output libraries that can run without API calls. It deliberately excludes cloud-based or retry-based frameworks like:

- Instructor
- LlamaIndex
- OpenAI's structured outputs
- Marvin
- Other API-based solutions

These frameworks are already thoroughly tested in the [original repository](https://github.com/stephenleo/llm-structured-output-benchmarks). The goal here is to evaluate libraries that:

1. Can run completely locally without internet connectivity
2. Structure outputs in a single pass without retry mechanisms
3. Are integrated more deeply with the LLM

This targeted scope allows us to better understand the performance characteristics and tradeoffs specific to these different structuring approaches.
