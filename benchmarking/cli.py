"""Command-line interface for the benchmarking pipeline."""

import argparse
import sys

from .llm_providers.base import LLMProvider
from .benchmarks import *
from .runner import BenchmarkRunner, BenchmarkRunConfig


def create_llm_provider(model_name: str, provider_type: str, system_prompt: str, **kwargs) -> LLMProvider:
    """Create an LLM provider based on the model name and type.
    
    Args:
        model_name (str): Name of the model
        provider_type (str): Type of provider (openai, google, openrouter, medrax)
        system_prompt (str): System prompt identifier to load from file
        **kwargs: Additional configuration parameters
        
    Returns:
        LLMProvider: The configured LLM provider
    """
    # Lazy imports to avoid slow startup
    if provider_type == "openai":
        from .llm_providers.openai_provider import OpenAIProvider
        provider_class = OpenAIProvider
    elif provider_type == "google":
        from .llm_providers.google_provider import GoogleProvider
        provider_class = GoogleProvider
    elif provider_type == "openrouter":
        from .llm_providers.openrouter_provider import OpenRouterProvider
        provider_class = OpenRouterProvider
    elif provider_type == "medrax":
        from .llm_providers.medrax_provider import MedRAXProvider
        provider_class = MedRAXProvider
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: openai, google, openrouter, medrax")
    
    return provider_class(model_name, system_prompt, **kwargs)


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
        "rexvqa": ReXVQABenchmark,
        "chestagentbench": ChestAgentBenchBenchmark,
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
    
    llm_provider = create_llm_provider(model_name=args.model, provider_type=args.provider, system_prompt=args.system_prompt, **provider_kwargs)
    
    # Create benchmark
    benchmark_kwargs = {}
    
    benchmark = create_benchmark(benchmark_name=args.benchmark, data_dir=args.data_dir, **benchmark_kwargs)
    
    # Create runner config
    config = BenchmarkRunConfig(
        provider_name=args.provider,
        model_name=args.model,
        benchmark_name=args.benchmark,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    summary = runner.run_benchmark(llm_provider, benchmark)
    
    print("\n" + "="*50)
    print("BENCHMARK COMPLETED")
    print("="*50)
    
    # Check if benchmark run was successful
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return
    
    # Print results
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Total Questions: {summary['results']['total_questions']}")
    print(f"Correct Answers: {summary['results']['correct_answers']}")
    print(f"Overall Accuracy: {summary['results']['accuracy']:.2f}%")
    print(f"Total Duration: {summary['results']['total_duration']:.2f}s")
    print(f"Results saved to: {summary['results_file']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MedRAX Benchmarking Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run benchmark command
    run_parser = subparsers.add_parser("run", help="Run a benchmark evaluation")
    run_parser.add_argument("--model", required=True, 
                           help="Model name (e.g., gpt-4o, gpt-4.1-2025-04-14, gemini-2.5-pro)")
    run_parser.add_argument("--provider", required=True, 
                           choices=["openai", "google", "openrouter", "medrax"], 
                           help="LLM provider to use")
    run_parser.add_argument("--system-prompt", required=True, 
                           choices=["MEDICAL_ASSISTANT", "CHESTAGENTBENCH_PROMPT"], 
                           help="System prompt: MEDICAL_ASSISTANT (general) or CHESTAGENTBENCH_PROMPT (benchmarks)")
    run_parser.add_argument("--benchmark", required=True, 
                           choices=["rexvqa", "chestagentbench"], 
                           help="Benchmark dataset: rexvqa (radiology VQA) or chestagentbench (chest X-ray reasoning)")
    run_parser.add_argument("--data-dir", required=True, 
                           help="Directory containing benchmark data files")
    run_parser.add_argument("--output-dir", default="benchmark_results", 
                           help="Output directory for results (default: benchmark_results)")
    run_parser.add_argument("--max-questions", type=int, 
                           help="Maximum number of questions to process (default: all)")
    run_parser.add_argument("--temperature", type=float, default=0.7, 
                           help="Model temperature for response generation (default: 0.7)")
    run_parser.add_argument("--top-p", type=float, default=0.95, 
                           help="Top-p nucleus sampling parameter (default: 0.95)")
    run_parser.add_argument("--max-tokens", type=int, default=5000, 
                           help="Maximum tokens per model response (default: 5000)")
    
    run_parser.set_defaults(func=run_benchmark_command)
    
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