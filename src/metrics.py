import itertools
from typing import Any
from dataclasses import dataclass, field, asdict
from loguru import logger

import numpy as np
import pandas as pd


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call comparison."""

    tool_name: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCallMetrics:
    """Comprehensive metrics for function calling evaluation."""

    call_structure: dict[str, Any] = field(
        default_factory=lambda: {
            "call_count_match": False,
            "expected_calls": 0,
            "predicted_calls": 0,
        }
    )
    tool_accuracy: dict[str, float] = field(
        default_factory=lambda: {"exact_match_ratio": 0.0, "total_calls": 0}
    )
    argument_coverage: dict[str, float] = field(
        default_factory=lambda: {
            "avg_matching_args_ratio": 0.0,
            "avg_extra_args_ratio": 0.0,
            "avg_missing_args_ratio": 0.0,
            "calls_with_perfect_coverage": 0,
        }
    )
    value_accuracy: dict[str, float] = field(
        default_factory=lambda: {
            "matching_value_ratio": 0.0,
            "perfect_value_matches": 0,
        }
    )
    overall_quality: dict[str, float] = field(
        default_factory=lambda: {"perfect_calls": 0, "weighted_score": 0.0}
    )
    detailed_metrics: list[dict[str, Any]] = field(default_factory=list)


def get_string_similarity(str1: str, str2: str) -> float:
    """Calculate string similarity using Levenshtein distance."""
    if not isinstance(str1, str) or not isinstance(str2, str):
        str1, str2 = str(str1), str(str2)

    # Case insensitive comparison
    str1, str2 = str1.lower(), str2.lower()

    if str1 == str2:
        return 1.0
    if len(str1) == 0 or len(str2) == 0:
        return 0.0

    # Calculate Levenshtein distance
    matrix = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

    for i in range(len(str1) + 1):
        matrix[i][0] = i
    for j in range(len(str2) + 1):
        matrix[0][j] = j

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + 1,  # substitution
                )

    max_len = max(len(str1), len(str2))
    similarity = 1 - (matrix[len(str1)][len(str2)] / max_len)
    return round(similarity, 3)


def analyze_tool_name(
    exp_call: dict[str, Any], pred_call: dict[str, Any]
) -> dict[str, Any]:
    """Analyze tool name matches between expected and predicted calls."""
    expected = exp_call["name"]
    predicted = pred_call["name"]
    similarity = get_string_similarity(expected, predicted)
    return {
        "expected": expected,
        "predicted": predicted,
        "matches": expected == predicted,
        "similarity": similarity,
    }


def analyze_arguments(
    exp_args: dict[str, Any], pred_args: dict[str, Any]
) -> dict[str, Any]:
    """Analyze argument matches between expected and predicted calls."""
    exp_keys = set(exp_args.keys())
    pred_keys = set(pred_args.keys())

    matching_args = exp_keys & pred_keys
    extra_args = pred_keys - exp_keys
    missing_args = exp_keys - pred_keys

    value_mismatches = {}
    matching_values = 0
    total_similarity = 0.0

    for arg in matching_args:
        if exp_args[arg] == pred_args[arg]:
            matching_values += 1
            total_similarity += 1.0
        else:
            similarity = get_string_similarity(str(exp_args[arg]), str(pred_args[arg]))
            total_similarity += similarity
            value_mismatches[arg] = {
                "expected": exp_args[arg],
                "predicted": pred_args[arg],
                "similarity": similarity,
            }

    return {
        "matching": list(matching_args),
        "extra": list(extra_args),
        "missing": list(missing_args),
        "value_mismatches": value_mismatches,
        "matching_values": matching_values,
        "avg_value_similarity": round(total_similarity / max(len(matching_args), 1), 3),
    }


def calculate_call_metrics(
    exp_call: dict[str, Any], pred_call: dict[str, Any]
) -> ToolCallMetrics:
    """Calculate metrics for a single tool call comparison."""
    tool_name = analyze_tool_name(exp_call, pred_call)
    arguments = analyze_arguments(exp_call["arguments"], pred_call["arguments"])
    return ToolCallMetrics(tool_name=tool_name, arguments=arguments)


def calculate_aggregate_metrics(
    call_metrics: list[ToolCallMetrics], total_calls: int
) -> dict[str, dict[str, float]]:
    """Calculate aggregate metrics across all tool calls."""
    if not total_calls:
        return {
            "tool_accuracy": {"exact_match_ratio": 0.0, "total_calls": 0},
            "argument_coverage": {
                "avg_matching_args_ratio": 0.0,
                "avg_extra_args_ratio": 0.0,
                "avg_missing_args_ratio": 0.0,
                "calls_with_perfect_coverage": 0,
            },
            "value_accuracy": {"matching_value_ratio": 0.0, "perfect_value_matches": 0},
            "overall_quality": {"perfect_calls": 0, "weighted_score": 0.0},
        }

    # Calculate tool accuracy
    tool_matches = sum(1 for m in call_metrics if m.tool_name["matches"])
    tool_accuracy = {
        "exact_match_ratio": tool_matches / total_calls,
        "total_calls": total_calls,
    }

    # Calculate argument coverage
    total_matching_ratio = 0
    total_extra_ratio = 0
    total_missing_ratio = 0
    perfect_coverage = 0
    perfect_values = 0
    perfect_calls = 0
    total_matching_values = 0
    total_possible_matches = 0

    for metric in call_metrics:
        args = metric.arguments
        exp_args_count = len(args["matching"]) + len(args["missing"])
        pred_args_count = len(args["matching"]) + len(args["extra"])

        matching_ratio = (
            len(args["matching"]) / exp_args_count if exp_args_count else 1.0
        )
        extra_ratio = len(args["extra"]) / pred_args_count if pred_args_count else 0.0
        missing_ratio = len(args["missing"]) / exp_args_count if exp_args_count else 0.0

        total_matching_ratio += matching_ratio
        total_extra_ratio += extra_ratio
        total_missing_ratio += missing_ratio

        is_perfect_coverage = not args["extra"] and not args["missing"]
        is_perfect_values = args["matching_values"] == len(args["matching"])

        if is_perfect_coverage:
            perfect_coverage += 1
        if is_perfect_values and args["matching"]:
            perfect_values += 1
        if metric.tool_name["matches"] and is_perfect_coverage and is_perfect_values:
            perfect_calls += 1

        total_matching_values += args["matching_values"]
        total_possible_matches += len(args["matching"])

    argument_coverage = {
        "avg_matching_args_ratio": total_matching_ratio / total_calls,
        "avg_extra_args_ratio": total_extra_ratio / total_calls,
        "avg_missing_args_ratio": total_missing_ratio / total_calls,
        "calls_with_perfect_coverage": perfect_coverage,
    }

    value_accuracy = {
        "matching_value_ratio": (
            total_matching_values / total_possible_matches
            if total_possible_matches > 0
            else 0.0
        ),
        "perfect_value_matches": perfect_values,
    }

    # Calculate weighted score
    tool_weight, args_weight, values_weight = 0.4, 0.3, 0.3
    weighted_score = (
        tool_weight * tool_accuracy["exact_match_ratio"]
        + args_weight * argument_coverage["avg_matching_args_ratio"]
        + values_weight * value_accuracy["matching_value_ratio"]
    )

    overall_quality = {"perfect_calls": perfect_calls, "weighted_score": weighted_score}

    return {
        "tool_accuracy": tool_accuracy,
        "argument_coverage": argument_coverage,
        "value_accuracy": value_accuracy,
        "overall_quality": overall_quality,
    }


def calculate_function_calling_metrics(
    expected: dict[str, Any], predicted: dict[str, Any]
) -> dict[str, Any]:
    """Calculate comprehensive metrics for function calling evaluation.

    The metrics are designed to capture both deterministic and semantic aspects:

    1. Tool Selection (τ):
       - Exact match: Binary score for correct tool selection
       - Semantic similarity: Continuous score for tool name similarity

    2. Schema Conformance (φ):
       - Structural: Ratio of correctly structured arguments
       - Type safety: Penalty for type violations and hallucinated parameters

    3. Parameter Coherence (ρ):
       - Exact matches: Binary accuracy for parameter values
       - Semantic similarity: Continuous score for value similarity
       - Hallucination penalty: Cost for generating non-schema parameters

    4. Composite Scores:
       - Deterministic (Δ): Weighted combination of exact matches
       - Semantic (Ψ): Weighted combination with similarity measures

    Returns:
        Dictionary containing detailed evaluation metrics
    """
    try:
        metrics = FunctionCallMetrics()

        # Tool name analysis with enhanced similarity
        tool_name_metrics = analyze_tool_name(expected, predicted)
        argument_metrics = analyze_arguments(
            expected["arguments"], predicted["arguments"]
        )

        # 1. Tool Selection Metrics
        exact_match = float(tool_name_metrics["matches"])
        name_similarity = tool_name_metrics["similarity"]

        # Calculate tool score with emphasis on exact matches
        tool_score = exact_match if exact_match else (name_similarity * 0.5)

        # 2. Schema Conformance
        matching_args = len(argument_metrics["matching"])
        total_expected = len(argument_metrics["matching"]) + len(
            argument_metrics["missing"]
        )
        total_predicted = len(argument_metrics["matching"]) + len(
            argument_metrics["extra"]
        )

        # Structural correctness
        arg_coverage = matching_args / total_expected if total_expected > 0 else 0.0

        # Type safety (penalize hallucination)
        type_safety = 1.0 - (
            len(argument_metrics["extra"]) / total_predicted
            if total_predicted > 0
            else 0.0
        )

        # Combined schema score
        schema_score = arg_coverage * (
            0.8 + 0.2 * type_safety
        )  # Base + type safety bonus

        # 3. Parameter Coherence
        matching_values = argument_metrics["matching_values"]
        total_matches = len(argument_metrics["matching"])

        # Exact value matches
        value_exact = matching_values / total_matches if total_matches else 0.0

        # Semantic similarity for values
        value_similarity = argument_metrics["avg_value_similarity"]

        # Hallucination penalty
        hallucination_penalty = (
            len(argument_metrics["extra"]) / total_predicted
            if total_predicted > 0
            else 0.0
        )

        # Combined value score
        value_score = (
            0.7 * value_exact  # Prioritize exact matches
            + 0.3 * value_similarity  # Consider semantic similarity
        ) * (1 - 0.2 * hallucination_penalty)  # Apply hallucination penalty

        # Package metrics
        metrics.tool_accuracy = {
            "exact_match_ratio": exact_match,
            "name_similarity": name_similarity,
            "tool_score": round(tool_score, 3),
            "total_calls": 1,
        }

        metrics.argument_coverage = {
            "matching_args_ratio": arg_coverage,
            "extra_args_ratio": len(argument_metrics["extra"]) / total_predicted
            if total_predicted
            else 0.0,
            "missing_args_ratio": len(argument_metrics["missing"]) / total_expected
            if total_expected
            else 0.0,
            "arg_score": round(schema_score, 3),
        }

        metrics.value_accuracy = {
            "exact_match_ratio": value_exact,
            "avg_similarity": value_similarity,
            "value_score": round(value_score, 3),
        }

        # Calculate composite scores
        deterministic_score = (
            0.4 * exact_match  # Tool selection is binary
            + 0.3 * arg_coverage  # Schema adherence is structural
            + 0.3 * value_exact  # Exact value matches
        )

        semantic_score = (
            0.4 * max(exact_match, name_similarity)  # Best of exact/semantic
            + 0.3 * schema_score  # Weighted schema score
            + 0.3 * value_score  # Value score with hallucination penalty
        )

        metrics.overall_quality = {
            "deterministic_score": round(deterministic_score, 3),
            "semantic_score": round(semantic_score, 3),
            "weighted_score": round(semantic_score, 3),  # Use semantic as primary
        }

        metrics.detailed_metrics = [
            {"tool_name": tool_name_metrics, "arguments": argument_metrics}
        ]
        return asdict(metrics)

    except Exception as e:
        logger.error(f"Error calculating function calling metrics: {str(e)}")
        logger.exception(e)
        return asdict(FunctionCallMetrics())


def reliability_metric(percent_successful: dict[str, list[float]]) -> pd.DataFrame:
    """Calculate reliability metrics across frameworks.

    Args:
        percent_successful: Dictionary mapping framework names to lists of success rates

    Returns:
        DataFrame with mean reliability scores for each framework
    """
    data = {k.replace("Framework", ""): v for k, v in percent_successful.items()}
    df = pd.DataFrame(data)

    reliability = pd.DataFrame(
        df.mean().values, index=df.mean().index, columns=["Reliability"]
    )
    return reliability.round(3).sort_values("Reliability", ascending=False)


def latency_metric(
    latencies: dict[str, list[list[float]]], percentile: int = 95
) -> pd.DataFrame:
    """Calculate latency percentiles across frameworks.

    Args:
        latencies: Dictionary mapping framework names to lists of latency measurements
        percentile: The percentile to calculate (default: 95)

    Returns:
        DataFrame with latency percentiles for each framework
    """
    # Flatten nested latency lists and calculate percentiles
    processed_latencies = {}
    for key, value in latencies.items():
        flat_latencies = list(itertools.chain(*value))
        if flat_latencies:
            framework_name = key.replace("Framework", "")
            processed_latencies[framework_name] = np.percentile(
                flat_latencies, percentile
            )

    latency_df = pd.DataFrame(
        list(processed_latencies.values()),
        index=list(processed_latencies.keys()),
        columns=[f"Latency_p{percentile}(s)"],
    )
    return latency_df.round(3).sort_values(f"Latency_p{percentile}(s)", ascending=True)


def variety_metric(predictions: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Calculate variety scores for function calling predictions."""
    if not predictions:
        return pd.DataFrame(columns=["Variety"])

    variety_scores = {}

    for framework, calls in predictions.items():
        framework_name = framework.replace("Framework", "")
        if not calls:
            variety_scores[framework_name] = 0.0
            continue

        unique_functions = len({call.get("name", "") for call in calls})
        variety_scores[framework_name] = unique_functions / len(calls)

    variety_df = pd.DataFrame(
        list(variety_scores.values()),
        index=list(variety_scores.keys()),
        columns=["Variety"],
    )
    return variety_df.round(3).sort_values("Variety", ascending=False)
