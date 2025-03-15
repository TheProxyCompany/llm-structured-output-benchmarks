# 🧩 LLM Structured Output Benchmarks

Benchmark various Structured Output frameworks:
    - Outlines
    - LMFormatEnforcer
    - Proxy Structuring Engine

This is a quick fork from the original [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) repository.

## 🏆 Benchmark Results [2025-02-26]

### Function Calling Results

| Metric | LMFormatEnforcerFramework | PSEFramework | OutlinesFramework |
|:------|:-------------------------:|:------------:|:-----------------:|
| Error Rate | 0.0% | 0.0% | 0.0% |
| Average Generation Time | 9.950s | **6.792s** | 15.835s |
| 95th Percentile Generation Time | 11.308s | **8.278s** | 17.290s |
| Average Correct Function Name | 83.0% | 83.0% | 82.0% |
| Average Correct Function Args | 78.8% | 78.4% | 79.5% |

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

1. Function Calling
    - **Task**: Evaluate the framework's ability to parse natural language into structured function calls
    - **Data**:
        - Sourced from fireworks-ai/function-calling-eval-dataset-v0
        - Contains prompts, expected function call completions, and available tools/functions
    - **Prompt**: The prompt is a user message that includes the available tools & definitions, and the user's query.
    - **Experiment Details**:
        - Run each sample through the framework `n_runs` times, using consistent random seeds across frameworks
        - Each nth run uses the same random seed across all frameworks to ensure reproducible comparisons
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
