import json
import openai
import os
import time
import logging
import base64
import requests
from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt
from datasets import load_dataset

# Initialize global variables
logger = logging.getLogger('benchmark')
model_name = 'chatgpt-4o-latest'  # default value
temperature = 0.2  # default value
log_filename = None

def setup_logging(filename):
    """Setup logging configuration"""
    global logger
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create file handler
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    return logger

def encode_image(image_path):
    """Encode local image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

def encode_image_url(image_url):
    """Encode image from URL to base64 string"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image from URL {image_url}: {str(e)}")
        return None

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def create_multimodal_request(example, client, use_urls=False, shutdown_event=None):
    """
    Create a multimodal request from a dataset example
    
    Args:
        example: Dataset example to process
        client: OpenAI client
        use_urls: Boolean flag to use image URLs instead of local files
        shutdown_event: Optional threading.Event for graceful shutdown
    """
    prompt = f"""Given the following medical case:
Please answer this multiple choice question:
{example['question']}
Base your answer only on the provided images and case information."""

    content = [{"type": "text", "text": prompt}]

    if use_urls:
        # Handle image URLs from the dataset
        image_urls = example['image_source_urls']
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        elif isinstance(image_urls[0], list):  # Handle nested lists
            image_urls = [url for sublist in image_urls for url in sublist]
        
        for img_url in image_urls:
            if img_url and isinstance(img_url, str):
                base64_image = encode_image_url(img_url)
                if base64_image:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                    print(f"Successfully loaded image from URL: {img_url}")
    else:
        # Handle local image files
        image_paths = example['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif isinstance(image_paths[0], list):  # Handle nested lists
            image_paths = [path for sublist in image_paths for path in sublist]
        
        for img_path in image_paths:
            if img_path and isinstance(img_path, str):
                img_path = img_path.replace('figures/', '')
                full_path = os.path.join("figures", img_path)
                
                if os.path.exists(full_path):
                    base64_image = encode_image(full_path)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                        print(f"Successfully loaded image: {full_path}")
                else:
                    print(f"Image file not found: {full_path}")

    # If no images found, log and return None
    if len(content) == 1:  # Only the text prompt exists
        print(f"No images found for question {example.get('question_id', 'unknown')}")
        log_entry = {
            "question_id": example.get('question_id', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "status": "skipped",
            "reason": "no_images",
            "input": {
                "question": example['question'],
                "explanation": example.get('explanation', ''),
                "image_paths": example.get('images' if not use_urls else 'image_source_urls')
            }
        }
        logger.info(json.dumps(log_entry))
        return None

    messages = [
        {"role": "system", "content": "You are a medical imaging expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."},
        {"role": "user", "content": content}
    ]

    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=50,
            temperature=temperature
        )
        duration = time.time() - start_time

        log_entry = {
            "question_id": example.get('question_id', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "duration": round(duration, 2),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model_answer": response.choices[0].message.content,
            "correct_answer": example['answer'],
            "input": {
                "messages": messages,
                "question": example['question'],
                "explanation": example.get('explanation', ''),
                "image_source": "url" if use_urls else "local",
                "images": example.get('image_source_urls' if use_urls else 'images')
            }
        }
        logger.info(json.dumps(log_entry))
        return response

    except Exception as e:
        log_entry = {
            "question_id": example.get('question_id', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "status": "error",
            "error": str(e),
            "input": {
                "messages": messages,
                "question": example['question'],
                "explanation": example.get('explanation', ''),
                "image_source": "url" if use_urls else "local",
                "images": example.get('image_source_urls' if use_urls else 'images')
            }
        }
        logger.info(json.dumps(log_entry))
        print(f"Error processing question {example.get('question_id', 'unknown')}: {str(e)}")
        raise

def main():
    import signal
    import threading
    import argparse
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run medical image analysis benchmark')
    parser.add_argument('--use-urls', action='store_true', help='Use image URLs instead of local files')
    parser.add_argument('--model', type=str, default='chatgpt-4o-latest', help='Model name to use')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for model inference')
    parser.add_argument('--log-prefix', type=str, help='Prefix for log filename (default: model name)')
    parser.add_argument('--max-cases', type=int, default=None, help='Maximum number of cases to process (default: all)')
    args = parser.parse_args()
    
    # Set global variables
    global model_name, temperature, log_filename
    model_name = args.model
    temperature = args.temperature
    log_prefix = args.log_prefix if args.log_prefix is not None else args.model
    log_filename = f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Setup logging
    setup_logging(log_filename)
    
    # Create an event for handling graceful shutdown
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        print("\nShutdown signal received. Completing current task...")
        shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("json", data_files="chestagentbench/metadata.jsonl")
    train_dataset = dataset["train"]

    # Collecting ENV variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    kwargs = {}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url

    # Initialize the OpenAI Client
    client = openai.OpenAI(api_key=api_key, **kwargs)

    total_examples = len(train_dataset)
    processed = 0
    skipped = 0

    print(f"Beginning benchmark evaluation for model {model_name}")
    print(f"Using {'image URLs' if args.use_urls else 'local files'} for images")
    print(f"Temperature: {temperature}")

    # Handle max cases limit
    dataset_to_process = train_dataset
    if args.max_cases is not None:
        dataset_to_process = train_dataset.select(range(min(args.max_cases, len(train_dataset))))
        total_examples = len(dataset_to_process)
        print(f"Processing {total_examples} cases (limited by --max-cases argument)")

    for example in dataset_to_process:
        if shutdown_event.is_set():
            print("\nGraceful shutdown initiated. Saving progress...")
            break
            
        processed += 1
        
        response = create_multimodal_request(example, client, args.use_urls, shutdown_event)

        if response is None:
            skipped += 1
            print(f"Skipped question: {example.get('question_id', 'unknown')}")
            continue

        print(f"Progress: {processed}/{total_examples}")
        print(f"Question ID: {example.get('question_id', 'unknown')}")
        print(f"Model Answer: {response.choices[0].message.content}")
        print(f"Correct Answer: {example['answer']}\n")

    print(f"\nBenchmark Summary:")
    print(f"Total Examples Processed: {processed}")
    print(f"Total Examples Skipped: {skipped}")
    
    # Verify log file exists and has content
    if os.path.exists(log_filename) and os.path.getsize(log_filename) > 0:
        print(f"\nLog file saved to: {os.path.abspath(log_filename)}")
    else:
        print(f"\nWarning: Log file could not be verified at: {os.path.abspath(log_filename)}")
        print("Please check directory permissions and available disk space.")

if __name__ == "__main__":
    main()