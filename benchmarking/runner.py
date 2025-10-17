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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    chunk_history: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkRunConfig:
    """Configuration for a benchmark run."""
    benchmark_name: str
    provider_name: str
    model_name: str
    output_dir: str
    max_questions: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 5000
    concurrency: int = 1
    random_seed: Optional[int] = None
    


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
        self.run_id = f"{config.benchmark_name}_{config.provider_name}_{config.model_name}_{config.max_questions}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
        benchmark: Benchmark,
        llm_provider: LLMProvider,
    ) -> Dict[str, Any]:
        """Run a benchmark against an LLM provider.
        
        Args:
            benchmark (Benchmark): The benchmark to run
            llm_provider (LLMProvider): The LLM provider to test
            
        Returns:
            Dict[str, Any]: Summary of benchmark results
        """
        self.logger.info(f"Starting benchmark run: {self.run_id}")
        self.logger.info(f"Benchmark: {benchmark}")
        self.logger.info(f"Provider: {llm_provider.provider_name}")
        self.logger.info(f"Model: {llm_provider.model_name}")
        
        # Test provider connection
        if not llm_provider.test_connection():
            self.logger.error("LLM provider connection test failed")
            return {"error": "LLM provider connection test failed"}
        
        # Initialize counters
        processed = 0
        correct = 0
        total_duration = 0.0
        
        # Determine concurrency
        max_workers = max(1, int(getattr(self.config, "concurrency", 1) or 1))
        
        # Process data points in parallel using a bounded thread pool
        with tqdm(total=len(benchmark), desc="Processing questions") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {executor.submit(self._process_data_point, dp, llm_provider): idx for idx, dp in enumerate(benchmark)}
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing data point {idx}: {e}")
                        result = BenchmarkResult(
                            data_point_id=f"error_{idx}",
                            question="",
                            model_answer="",
                            correct_answer="",
                            is_correct=False,
                            duration=0.0,
                            error=str(e)
                        )
                    
                    # Update counters
                    processed += 1
                    if result.is_correct:
                        correct += 1
                    total_duration += result.duration
                    
                    # Add to results and persist immediately
                    self.results.append(result)
                    self._save_individual_result(result)
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Periodic logging
                    if processed % 10 == 0:
                        accuracy = (correct / processed) * 100
                        avg_duration = total_duration / processed if processed > 0 else 0.0
                        self.logger.info(
                            f"Progress: {processed}/{len(benchmark)} | "
                            f"Accuracy: {accuracy:.2f}% | "
                            f"Avg Duration: {avg_duration:.2f}s"
                        )
        
        # Save final results
        summary = self._save_final_results(benchmark)

        self.logger.info(f"Benchmark run completed: {self.run_id}")
        self.logger.info(f"Summary: {summary}")

        return summary

    def _process_data_point(
        self,
        data_point: BenchmarkDataPoint,
        llm_provider: LLMProvider
    ) -> BenchmarkResult:
        """Process a single data point.
        
        Args:
            data_point (BenchmarkDataPoint): The data point to process
            llm_provider (LLMProvider): The LLM provider to use
            
        Returns:
            BenchmarkResult: Result of processing the data point
        """
        start_time = time.time()
        
        try:
            # Create request for LLM
            request = LLMRequest(
                text=data_point.text,
                images=data_point.images
            )
            
            # Get response from LLM
            response: LLMResponse = llm_provider.generate_response(request)
            
            # Extract answer (this may need customization based on benchmark)
            model_answer = self._extract_answer(response.content)
            
            # Check if correct
            is_correct = model_answer == data_point.correct_answer
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Return result
            return BenchmarkResult(
                data_point_id=data_point.id,
                question=data_point.text,
                model_answer=model_answer,
                correct_answer=data_point.correct_answer,
                is_correct=is_correct,
                duration=duration,
                usage=response.usage,
                chunk_history=response.chunk_history,
                metadata={
                    "data_point_metadata": data_point.metadata,
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
                chunk_history=None,
                metadata={
                    "data_point_metadata": data_point.metadata
                }
            )

    def _extract_answer(self, response_text: str) -> str:
        """Extract the answer from the model response.
        
        Args:
            response_text (str): The full response text from the model
            
        Returns:
            str: The extracted answer
        """
        # Look for the '\boxed{A}' format
        boxed_pattern = r'\\boxed\{([A-Fa-f])\}'
        match = re.search(boxed_pattern, response_text)
        if match:
            return match.group(1).upper()
        
        # If no pattern matches, return the full response
        return response_text.strip()

    def _save_individual_result(self, result: BenchmarkResult) -> None:
        """Save a single result to its own JSON file.
        
        Args:
            result (BenchmarkResult): The result to save
        """
        # Sanitize data_point_id for filename (remove invalid characters)
        safe_id = re.sub(r'[^\w\-_.]', '_', result.data_point_id)
        
        # Create run_id directory and individual_results subdirectory
        run_dir = self.output_dir / self.run_id
        individual_results_dir = run_dir / "individual_results"
        individual_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with benchmark name and data point ID
        filename = f"{self.config.benchmark_name}_{safe_id}.json"
        result_file = individual_results_dir / filename
        
        # Convert result to serializable format
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "data_point_id": result.data_point_id,
            "question": result.question,
            "model_answer": result.model_answer,
            "correct_answer": result.correct_answer,
            "is_correct": result.is_correct,
            "duration": result.duration,
            "usage": result.usage,
            "error": result.error,
            "chunk_history": result.chunk_history,
            "metadata": result.metadata,
        }
        
        # Save to file
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

    def _save_final_results(self, benchmark: Benchmark) -> Dict[str, Any]:
        """Save final results and return summary.
        
        Args:
            benchmark (Benchmark): The benchmark that was run
            
        Returns:
            Dict[str, Any]: Summary of results
        """
        # Create run_id directory and final_results subdirectory
        run_dir = self.output_dir / self.run_id
        final_results_dir = run_dir / "final_results"
        final_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = final_results_dir / f"{self.run_id}_results.json"
        
        # Convert results to serializable format for final file
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
        
        # Calculate summary statistics
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        total_duration = sum(r.duration for r in self.results)
        
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Create summary
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "benchmark_name": self.config.benchmark_name,
                "provider_name": self.config.provider_name,
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
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
            },
            "results_file": str(results_file),
        }
        
        # Save summary
        summary_file = final_results_dir / f"{self.run_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary 