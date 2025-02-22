import json
import random
from collections import Counter

import pandas as pd
from datasets import load_dataset
from loguru import logger
from rich.progress import track
from typer import Option, Typer

app = Typer()


def download_default_classification_dataset(
    text_column: str = "utt", label_column: str = "intent"
) -> pd.DataFrame:
    """Download the default classification dataset from Hugging Face.

    Uses the AmazonScience/massive dataset, which contains utterances and their associated intents.
    The dataset is transformed into a format suitable for classification tasks.

    Args:
        text_column: Column name for text data in source dataset. Defaults to "utt".
        label_column: Column name for labels in source dataset. Defaults to "intent".

    Returns:
        DataFrame with columns:
            - text (str): The utterance text
            - label (str): The intent/class label name
    """
    logger.info("Downloading source data from AmazonScience/massive")
    dataset = load_dataset("AmazonScience/massive", "en-US", split="test")
    dataset = dataset.select_columns([text_column, label_column])

    logger.info("Processing text and label columns")
    dataset = dataset.rename_columns({text_column: "text", label_column: "class_label"})
    class_names = dataset.features["class_label"].names  # type: ignore

    dataset = dataset.map(
        lambda row: {"label": class_names[row["class_label"]]},
        remove_columns=["class_label"],
    )

    return dataset.to_pandas()  # type: ignore


@app.command()
def generate_multilabel_data(
    source_data_pickle_path: str = Option(
        None,
        help="Path to source DataFrame pickle. Must have text and labels columns.",
    ),
    source_dataframe_text_column: str = Option(
        "text", help="Column name for text data."
    ),
    source_dataframe_label_column: str = Option(
        "label", help="Column name for labels."
    ),
    dest_num_rows: int = Option(100, help="Number of rows in final DataFrame."),
    dest_label_distribution: str = Option(
        default='{"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15}',
        help="JSON string of probability for number of entities per row.",
    ),
) -> None:
    """Generate synthetic multilabel classification data by combining source rows.

    This function creates a synthetic dataset for multilabel classification by:
    1. Taking the source single-label dataset (either provided or downloaded)
    2. Randomly sampling N rows according to the label distribution
    3. Combining their texts with periods and merging their labels

    The resulting dataset structure is:
    {
        "text": [str],       # Combined utterances from source rows
        "labels": [list],    # List of labels, one list per row containing multiple labels
    }

    Example output row:
    {
        "text": "Turn on the lights. What's the weather like?",
        "labels": ["SmartHome.Light", "Weather.Query"]
    }

    Args:
        source_data_pickle_path: Path to source DataFrame pickle. If None, downloads default dataset.
        source_dataframe_text_column: Column name for text data in source DataFrame.
        source_dataframe_label_column: Column name for labels in source DataFrame.
        dest_num_rows: Number of rows to generate in final DataFrame.
        dest_label_distribution: JSON string mapping number of labels to probability.
            Example: {"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15} means:
            - 35% of rows will have 1 label
            - 30% of rows will have 2 labels
            - 20% of rows will have 3 labels
            - 15% of rows will have 4 labels
    """
    dest_label_distribution_dict: dict[str, float] = json.loads(dest_label_distribution)
    dest_labels: dict[int, float] = {
        int(k): v for k, v in dest_label_distribution_dict.items()
    }

    if not source_data_pickle_path:
        logger.info("No source pickle provided, downloading default dataset")
        source_dataframe = download_default_classification_dataset()
    else:
        logger.info("Loading source data from pickle")
        source_dataframe = pd.read_pickle(source_data_pickle_path)

    logger.info(f"Generating {dest_num_rows} synthetic rows")

    multilabel_data = {"text": [], "labels": []}
    for _ in track(range(dest_num_rows), description="Generating rows"):
        num_rows = random.choices(list(dest_labels.keys()), list(dest_labels.values()))[
            0
        ]
        random_rows = source_dataframe.sample(num_rows)

        multilabel_data["text"].append(
            ". ".join(random_rows[source_dataframe_text_column].tolist())
        )
        multilabel_data["labels"].append(
            random_rows[source_dataframe_label_column].tolist()
        )

    multilabel_df = pd.DataFrame(multilabel_data)

    label_counter = Counter([len(label) for label in multilabel_df["labels"]])
    label_counter = pd.DataFrame.from_records(
        list(label_counter.items()), columns=["num_labels", "num_rows"]
    ).sort_values("num_labels")

    logger.info(f"Number of rows per label count:\n{label_counter.head()}")
    logger.info(f"First 5 rows:\n{multilabel_df.head()}")

    multilabel_df.to_pickle("data/multilabel_classification.pkl")
    logger.info("Saved multilabel data to: data/multilabel_classification.pkl")


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
