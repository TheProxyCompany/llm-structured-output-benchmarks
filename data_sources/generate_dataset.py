import pandas as pd
from datasets import load_dataset
from loguru import logger
from typer import Option, Typer

app = Typer()

def download_function_calling_dataset() -> pd.DataFrame:
    """Download function calling dataset from Hugging Face.

    Uses fireworks-ai/function-calling-eval-dataset-v0, which contains examples of
    function calling scenarios with prompts and expected completions.

    Returns:
        DataFrame with columns:
            - prompt (str): The input prompt/query
            - completion (str): The expected function call completion
            - tools (list): Available tools/functions that can be called
    """
    logger.info("Downloading function calling evaluation dataset")
    dataset = load_dataset(
        "fireworks-ai/function-calling-eval-dataset-v0", split="single_turn"
    )
    dataset = dataset.select_columns(["prompt", "completion", "tools"])
    return dataset.to_pandas()  # type: ignore

@app.command()
def generate_function_calling_data(
    dest_num_rows: int = Option(100, help="Number of rows in final DataFrame."),
) -> None:
    """Generate function calling data by sampling source dataset rows.

    This function creates a dataset for evaluating function calling capabilities by
    sampling from the fireworks-ai/function-calling-eval-dataset-v0 dataset.

    The resulting dataset structure is identical to the source:
    {
        "prompt": [str],      # Input prompts/queries
        "completion": [str],  # Expected function call completions
        "tools": [list]      # Available tools/functions for each scenario
    }
    """
    source_dataframe = download_function_calling_dataset()
    logger.info(f"Generating {dest_num_rows} synthetic rows")

    function_calling_df = source_dataframe.sample(
        n=dest_num_rows, random_state=1
    ).reset_index(drop=True)

    logger.info(f"First 5 rows:\n{function_calling_df.head()}")
    function_calling_df.to_pickle("data/function_calling.pkl")
    logger.info("Saved function calling data to: data/function_calling.pkl")


if __name__ == "__main__":
    app()
