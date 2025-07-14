"""Evaluation code for analyzing benchmark results."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class EvaluationResult:
    """Results of evaluating a benchmark run."""
    model_name: str
    benchmark_name: str
    overall_accuracy: float
    total_questions: int
    correct_answers: int
    total_duration: float
    category_accuracies: Dict[str, float]
    category_counts: Dict[str, int]
    error_rate: float
    avg_duration_per_question: float


class BenchmarkEvaluator:
    """Class for evaluating and comparing benchmark results."""

    def __init__(self, output_dir: str = "evaluation_results"):
        """Initialize the evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load benchmark results from file.
        
        Args:
            results_file (str): Path to the results file
            
        Returns:
            Dict[str, Any]: Loaded results data
        """
        with open(results_file, 'r') as f:
            return json.load(f)

    def evaluate_single_run(self, results_file: str) -> EvaluationResult:
        """Evaluate a single benchmark run.
        
        Args:
            results_file (str): Path to the results file
            
        Returns:
            EvaluationResult: Evaluation results
        """
        results = self.load_results(results_file)
        
        # Calculate basic metrics
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.get("is_correct", False))
        total_duration = sum(r.get("duration", 0) for r in results)
        errors = sum(1 for r in results if r.get("error") is not None)
        
        overall_accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        error_rate = (errors / total_questions) * 100 if total_questions > 0 else 0
        
        # Calculate per-category metrics
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in results:
            metadata = result.get("metadata", {})
            category = metadata.get("category")
            
            if category:
                category_stats[category]["total"] += 1
                if result.get("is_correct", False):
                    category_stats[category]["correct"] += 1
        
        # Calculate category accuracies
        category_accuracies = {}
        category_counts = {}
        for category, stats in category_stats.items():
            category_accuracies[category] = (stats["correct"] / stats["total"]) * 100
            category_counts[category] = stats["total"]
        
        # Extract model and benchmark names (assuming they're in the filename or metadata)
        results_path = Path(results_file)
        filename_parts = results_path.stem.split("_")
        
        model_name = "unknown"
        benchmark_name = "unknown"
        
        if len(filename_parts) >= 2:
            benchmark_name = filename_parts[0]
            model_name = filename_parts[1]
        
        return EvaluationResult(
            model_name=model_name,
            benchmark_name=benchmark_name,
            overall_accuracy=overall_accuracy,
            total_questions=total_questions,
            correct_answers=correct_answers,
            total_duration=total_duration,
            category_accuracies=category_accuracies,
            category_counts=category_counts,
            error_rate=error_rate,
            avg_duration_per_question=total_duration / total_questions if total_questions > 0 else 0,
        )
