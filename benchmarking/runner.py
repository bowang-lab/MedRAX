"""Main test runner for benchmarking pipeline."""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import re
from .llm_providers import LLMProvider, LLMRequest, LLMResponse
from .benchmarks import Benchmark, BenchmarkDataPoint


@dataclass
class BenchmarkResult:
    """Result of running a benchmark on a single data point."""
    data_point_id: str
    question: str
    model_answer: str
    correct_answer: str
    is_correct: bool
    duration: float
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkRunConfig:
    """Configuration for a benchmark run."""
    provider_name: str
    model_name: str
    benchmark_name: str
    output_dir: str
    max_questions: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 5000
    additional_params: Optional[Dict[str, Any]] = None


class BenchmarkRunner:
    """Main class for running benchmarks against LLM providers."""

    def __init__(self, config: BenchmarkRunConfig):
        """Initialize the benchmark runner.
        
        Args:
            config (BenchmarkRunConfig): Configuration for the benchmark run
        """
        self.config = config
        self.results = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique run ID
        self.run_id = f"{config.benchmark_name}_{config.provider_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up logging
        self._setup_logging()
        
        self.logger.info(f"Initialized benchmark runner with ID: {self.run_id}")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.output_dir / f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create logger
        self.logger = logging.getLogger(f"benchmark_runner_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def run_benchmark(
        self,
        llm_provider: LLMProvider,
        benchmark: Benchmark,
    ) -> Dict[str, Any]:
        """Run a benchmark against an LLM provider.
        
        Args:
            llm_provider (LLMProvider): The LLM provider to test
            benchmark (Benchmark): The benchmark to run
            
        Returns:
            Dict[str, Any]: Summary of benchmark results
        """
        self.logger.info(f"Starting benchmark run: {self.run_id}")
        self.logger.info(f"Model: {llm_provider.model_name}")
        self.logger.info(f"Benchmark: {benchmark}")
        
        # Test provider connection
        if not llm_provider.test_connection():
            self.logger.error("LLM provider connection test failed")
            return {"error": "LLM provider connection test failed"}
        
        # Get data points to process
        total_questions = len(benchmark)
        max_questions = self.config.max_questions or total_questions
        end_index = min(max_questions, total_questions)
        
        self.logger.info(f"Processing questions {0} to {end_index-1} of {total_questions}")
        
        # Initialize counters
        processed = 0
        correct = 0
        total_duration = 0.0
        
        # Process each data point
        for i in tqdm(range(0, end_index), desc="Processing questions"):
            try:
                data_point = benchmark.get_data_point(i)
                
                # Run the model on this data point
                result = self._process_data_point(llm_provider, data_point)
                
                # Update counters
                processed += 1
                if result.is_correct:
                    correct += 1
                total_duration += result.duration
                
                # Add to results
                self.results.append(result)
                
                # Log progress
                if processed % 10 == 0:
                    self._save_intermediate_results()
                    accuracy = (correct / processed) * 100
                    avg_duration = total_duration / processed
                    
                    self.logger.info(
                        f"Progress: {processed}/{end_index} | "
                        f"Accuracy: {accuracy:.2f}% | "
                        f"Avg Duration: {avg_duration:.2f}s"
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing data point {i}: {e}")
                # Add error result
                error_result = BenchmarkResult(
                    data_point_id=f"error_{i}",
                    question="",
                    model_answer="",
                    correct_answer="",
                    is_correct=False,
                    duration=0.0,
                    error=str(e)
                )
                self.results.append(error_result)
                continue
        
        # Save final results
        summary = self._save_final_results(benchmark)
        
        self.logger.info(f"Benchmark run completed: {self.run_id}")
        self.logger.info(f"Final accuracy: {summary['results']['accuracy']:.2f}%")
        self.logger.info(f"Total duration: {summary['results']['total_duration']:.2f}s")
        
        return summary

    def _process_data_point(
        self,
        llm_provider: LLMProvider,
        data_point: BenchmarkDataPoint,
    ) -> BenchmarkResult:
        """Process a single data point.
        
        Args:
            llm_provider (LLMProvider): The LLM provider to use
            data_point (BenchmarkDataPoint): The data point to process
            
        Returns:
            BenchmarkResult: Result of processing the data point
        """
        start_time = time.time()
        
        try:
            # Create request
            request = LLMRequest(
                text=data_point.text,
                images=data_point.images,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                additional_params=self.config.additional_params
            )
            
            # Get response from LLM
            response: LLMResponse = llm_provider.generate_response(request)
            
            # Extract answer (this may need customization based on benchmark)
            model_answer = self._extract_answer(response.content)
            
            # Check if correct
            is_correct = self._is_correct_answer(model_answer, data_point.correct_answer)
            
            duration = time.time() - start_time
            
            return BenchmarkResult(
                data_point_id=data_point.id,
                question=data_point.text,
                model_answer=model_answer,
                correct_answer=data_point.correct_answer,
                is_correct=is_correct,
                duration=duration,
                usage=response.usage,
                metadata={
                    "data_point_metadata": data_point.metadata,
                    "case_id": data_point.case_id,
                    "category": data_point.category,
                    "raw_response": response.content,
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                data_point_id=data_point.id,
                question=data_point.text,
                model_answer="",
                correct_answer=data_point.correct_answer,
                is_correct=False,
                duration=duration,
                error=str(e),
                metadata={
                    "data_point_metadata": data_point.metadata,
                    "case_id": data_point.case_id,
                    "category": data_point.category,
                }
            )

    def _extract_answer(self, response_text: str) -> str:
        """Extract the answer from the model response.
        
        Args:
            response_text (str): The full response text from the model
            
        Returns:
            str: The extracted answer
        """
        # First, look for the 'Final answer: <|A|>' format
        final_answer_pattern = r'Final answer:\s*<\|([A-F])\|>'
        match = re.search(final_answer_pattern, response_text)
        if match:
            return match.group(1).upper()

        # This is a simple implementation - may need customization per benchmark
        # For multiple choice, look for single letters A, B, C, D, E, F
        patterns = [
            r'\b([A-F])\b',  # Single letter
            r'\b([A-F])\)',  # Letter with closing parenthesis
            r'\(([A-F])\)',  # Letter in parentheses
            r'[Aa]nswer\s*:?\s*([A-F])',  # "Answer: X" format
            r'[Cc]hoice\s*:?\s*([A-F])',  # "Choice: X" format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, return the first letter found
        letters = re.findall(r'\b[A-F]\b', response_text)
        if letters:
            return letters[0].upper()
        
        # If no letters found, return the full response (truncated)
        return response_text.strip()[:100]

    def _is_correct_answer(self, model_answer: str, correct_answer: str) -> bool:
        """Check if the model answer is correct.
        
        Args:
            model_answer (str): The model's answer
            correct_answer (str): The correct answer
            
        Returns:
            bool: True if the answer is correct
        """
        if not model_answer or not correct_answer:
            return False
        
        # For multiple choice, compare just the letter
        model_clean = model_answer.strip().upper()
        correct_clean = correct_answer.strip().upper()
        
        # Extract just the first letter for comparison
        model_letter = model_clean[0] if model_clean else ""
        correct_letter = correct_clean[0] if correct_clean else ""
        
        return model_letter == correct_letter

    def _save_intermediate_results(self) -> None:
        """Save intermediate results to disk."""
        results_file = self.output_dir / f"{self.run_id}_intermediate.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            results_data.append({
                "data_point_id": result.data_point_id,
                "question": result.question,
                "model_answer": result.model_answer,
                "correct_answer": result.correct_answer,
                "is_correct": result.is_correct,
                "duration": result.duration,
                "usage": result.usage,
                "error": result.error,
                "metadata": result.metadata,
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

    def _save_final_results(self, benchmark: Benchmark) -> Dict[str, Any]:
        """Save final results and return summary.
        
        Args:
            benchmark (Benchmark): The benchmark that was run
            
        Returns:
            Dict[str, Any]: Summary of results
        """
        # Save detailed results
        results_file = self.output_dir / f"{self.run_id}_results.json"
        self._save_intermediate_results()
        
        # Calculate summary statistics
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        total_duration = sum(r.duration for r in self.results)
        
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Calculate per-category accuracy
        category_stats = {}
        for result in self.results:
            if result.metadata and result.metadata.get("category"):
                category = result.metadata["category"]
                if category not in category_stats:
                    category_stats[category] = {"correct": 0, "total": 0}
                category_stats[category]["total"] += 1
                if result.is_correct:
                    category_stats[category]["correct"] += 1
        
        # Calculate accuracy for each category
        category_accuracies = {}
        for category, stats in category_stats.items():
            category_accuracies[category] = (stats["correct"] / stats["total"]) * 100
        
        # Create summary
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.model_name,
                "benchmark_name": self.config.benchmark_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            "benchmark_info": {
                "total_size": len(benchmark),
                "processed_questions": total_questions,
            },
            "results": {
                "accuracy": accuracy,
                "correct_answers": correct_answers,
                "total_questions": total_questions,
                "total_duration": total_duration,
                "avg_duration_per_question": total_duration / total_questions if total_questions > 0 else 0,
                "category_accuracies": category_accuracies,
            },
            "results_file": str(results_file),
        }
        
        # Save summary
        summary_file = self.output_dir / f"{self.run_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary 