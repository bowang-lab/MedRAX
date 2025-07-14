"""Command-line interface for the benchmarking pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .llm_providers import *
from .benchmarks import *
from .runner import BenchmarkRunner, BenchmarkRunConfig
from .evaluation import BenchmarkEvaluator


def create_llm_provider(model_name: str, provider_type: str, **kwargs) -> LLMProvider:
    """Create an LLM provider based on the model name and type.
    
    Args:
        model_name (str): Name of the model
        provider_type (str): Type of provider (openai, google, openrouter, medrax)
        **kwargs: Additional configuration parameters
        
    Returns:
        LLMProvider: The configured LLM provider
    """
    provider_map = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "openrouter": OpenRouterProvider,
        "medrax": MedRAXProvider,
    }
    
    if provider_type not in provider_map:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {list(provider_map.keys())}")
    
    provider_class = provider_map[provider_type]
    return provider_class(model_name, **kwargs)


def create_benchmark(benchmark_name: str, data_dir: str, **kwargs) -> Benchmark:
    """Create a benchmark based on the benchmark name.
    
    Args:
        benchmark_name (str): Name of the benchmark
        data_dir (str): Directory containing benchmark data
        **kwargs: Additional configuration parameters
        
    Returns:
        Benchmark: The configured benchmark
    """
    benchmark_map = {
        "chest_agent_bench": ChestAgentBench,
        "rexvqa": ReXVQABenchmark,
    }
    
    if benchmark_name not in benchmark_map:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(benchmark_map.keys())}")
    
    benchmark_class = benchmark_map[benchmark_name]
    return benchmark_class(data_dir, **kwargs)


def run_benchmark_command(args) -> None:
    """Run a benchmark."""
    print(f"Running benchmark: {args.benchmark} with model: {args.model}")
    
    # Create LLM provider
    provider_kwargs = {}
    if args.provider == "medrax":
        provider_kwargs = {
            "tools_to_use": args.medrax_tools.split(",") if args.medrax_tools else None,
            "model_dir": args.model_dir,
            "temp_dir": args.temp_dir,
            "device": args.device,
            "rag_config": None,  # You might want to add RAG config options
        }
    
    llm_provider = create_llm_provider(args.model, args.provider, **provider_kwargs)
    
    # Create benchmark
    benchmark_kwargs = {}
    
    benchmark = create_benchmark(args.benchmark, args.data_dir, **benchmark_kwargs)
    
    # Create runner config
    config = BenchmarkRunConfig(
        model_name=args.model,
        benchmark_name=args.benchmark,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        start_index=args.start_index,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        save_frequency=args.save_frequency,
        log_level=args.log_level,
    )
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    summary = runner.run_benchmark(llm_provider, benchmark)
    
    print("\n" + "="*50)
    print("BENCHMARK COMPLETED")
    print("="*50)
    print(f"Overall Accuracy: {summary['results']['accuracy']:.2f}%")
    print(f"Total Questions: {summary['results']['total_questions']}")
    print(f"Correct Answers: {summary['results']['correct_answers']}")
    print(f"Total Duration: {summary['results']['total_duration']:.2f}s")
    print(f"Results saved to: {summary['results_file']}")


def evaluate_results_command(args) -> None:
    """Evaluate benchmark results."""
    print(f"Evaluating results: {args.results_files}")
    
    evaluator = BenchmarkEvaluator(args.output_dir)
    
    if len(args.results_files) == 1:
        # Single model evaluation
        evaluation = evaluator.evaluate_single_run(args.results_files[0])
        print("\n" + "="*50)
        print("SINGLE MODEL EVALUATION")
        print("="*50)
        print(f"Model: {evaluation.model_name}")
        print(f"Benchmark: {evaluation.benchmark_name}")
        print(f"Overall Accuracy: {evaluation.overall_accuracy:.2f}%")
        print(f"Total Questions: {evaluation.total_questions}")
        print(f"Error Rate: {evaluation.error_rate:.2f}%")
        print(f"Total Duration: {evaluation.total_duration:.2f}s")
        
        if evaluation.category_accuracies:
            print("\nCategory Accuracies:")
            for category, accuracy in evaluation.category_accuracies.items():
                print(f"  {category}: {accuracy:.2f}%")
    
    else:
        # Multiple model comparison
        comparison = evaluator.compare_models(args.results_files)
        
        if "error" in comparison:
            print(f"Error: {comparison['error']}")
            return
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        summary = comparison["summary"]
        print(f"Models Compared: {summary['models_compared']}")
        print(f"Best Overall Accuracy: {summary['best_overall_accuracy']:.2f}%")
        print(f"Accuracy Range: {summary['accuracy_range'][0]:.2f}% - {summary['accuracy_range'][1]:.2f}%")
        
        best_model = comparison["best_model"]
        print(f"\nBest Model: {best_model['Model']} ({best_model['Accuracy (%)']:.2f}%)")
        
        # Generate comprehensive report
        report_path = evaluator.generate_report(args.results_files, args.report_name)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Statistical significance test
        if args.statistical_test:
            print("\nRunning statistical significance tests...")
            sig_results = evaluator.statistical_significance_test(args.results_files)
            print(f"Found {len(sig_results['comparisons'])} pairwise comparisons")
            
            for comp in sig_results["comparisons"]:
                significance = "significant" if comp["significant"] else "not significant"
                print(f"{comp['model1']} vs {comp['model2']}: {significance} (p={comp['p_value']:.4f})")


def list_providers_command(args) -> None:
    """List available LLM providers."""
    print("Available LLM Providers:")
    print("- openai: OpenAI GPT models")
    print("- google: Google Gemini models")
    print("- openrouter: OpenRouter API (multiple models)")
    print("- medrax: MedRAX agent system")


def list_benchmarks_command(args) -> None:
    """List available benchmarks."""
    print("Available Benchmarks:")
    print("- rexvqa: ReXVQA (large-scale chest radiology VQA)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MedRAX Benchmarking Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run benchmark command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    run_parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4o, gemini-2.5-pro)")
    run_parser.add_argument("--provider", required=True, choices=["openai", "google", "openrouter", "medrax"], help="LLM provider")
    run_parser.add_argument("--benchmark", required=True, choices=["chest_agent_bench", "rexvqa"], help="Benchmark to run")
    run_parser.add_argument("--data-dir", required=True, help="Directory containing benchmark data")
    run_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    run_parser.add_argument("--max-questions", type=int, help="Maximum number of questions to process")
    run_parser.add_argument("--start-index", type=int, default=0, help="Starting index for questions")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
    run_parser.add_argument("--max-tokens", type=int, default=1500, help="Maximum tokens per response")
    run_parser.add_argument("--system-prompt", help="System prompt for the model")
    run_parser.add_argument("--save-frequency", type=int, default=10, help="Save results every N questions")
    run_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # MedRAX-specific arguments
    run_parser.add_argument("--medrax-tools", help="Comma-separated list of tools for MedRAX (e.g., WebBrowserTool,MedicalRAGTool)")
    run_parser.add_argument("--model-dir", default="/model-weights", help="Directory containing model weights for MedRAX")
    run_parser.add_argument("--temp-dir", default="temp", help="Temporary directory for MedRAX")
    run_parser.add_argument("--device", default="cuda", help="Device for MedRAX models")
    

    
    run_parser.set_defaults(func=run_benchmark_command)
    
    # Evaluate results command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate benchmark results")
    eval_parser.add_argument("results_files", nargs="+", help="Path(s) to results files")
    eval_parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for evaluation")
    eval_parser.add_argument("--report-name", default="evaluation_report", help="Name for the evaluation report")
    eval_parser.add_argument("--statistical-test", action="store_true", help="Run statistical significance tests")
    eval_parser.set_defaults(func=evaluate_results_command)
    
    # List providers command
    list_providers_parser = subparsers.add_parser("list-providers", help="List available LLM providers")
    list_providers_parser.set_defaults(func=list_providers_command)
    
    # List benchmarks command
    list_benchmarks_parser = subparsers.add_parser("list-benchmarks", help="List available benchmarks")
    list_benchmarks_parser.set_defaults(func=list_benchmarks_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 