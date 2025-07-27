import json
import random
from pathlib import Path
from typing import Dict, Optional, Any
from .base import Benchmark, BenchmarkDataPoint

class ChestAgentBenchBenchmark(Benchmark):
    """ChestAgentBench benchmark for complex CXR interpretation and reasoning.
    
    Loads the dataset from a local metadata.jsonl file and parses each entry into a BenchmarkDataPoint.
    """
    def __init__(self, data_dir: str, **kwargs):
        self.max_questions = kwargs.get("max_questions", None)
        super().__init__(data_dir, **kwargs)

    def _load_data(self) -> None:
        metadata_path = Path(self.data_dir) / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Could not find metadata.jsonl in {self.data_dir}")
        print(f"Loading ChestAgentBench from local file: {metadata_path}")
        self.data_points = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.max_questions and i >= self.max_questions:
                    break
                try:
                    item = json.loads(line)
                    data_point = self._parse_item(item, i)
                    if data_point:
                        self.data_points.append(data_point)
                except Exception as e:
                    print(f"Error loading item {i}: {e}")
                    continue
        
        # Shuffle the final data
        random.shuffle(self.data_points, seed=42)

    def _parse_item(self, item: Dict[str, Any], index: int) -> Optional[BenchmarkDataPoint]:
        # Use full_question_id or question_id if available, else fallback
        question_id = item.get("full_question_id") or item.get("question_id") or f"chestagentbench_{index}"
        question = item.get("question", "")
        correct_answer = item.get("answer", "")
        explanation = item.get("explanation", "")
        images = item.get("images", [])
        case_id = item.get("case_id", "")
        category = item.get("categories", "")
        # Compose question text (options are embedded in the question string)
        question_with_options = question
        # Map image paths to local figures directory
        local_images = None
        if images:
            figures_dir = Path(self.data_dir) / "figures"
            local_images = []
            for img in images:
                # Handle relative paths like "figures/11583/figure_1.jpg"
                if img.startswith("figures/"):
                    # Remove "figures/" prefix and construct full path
                    relative_path = img[8:]  # Remove "figures/" prefix
                    full_path = figures_dir / relative_path
                    local_images.append(str(full_path))
                else:
                    # Fallback to original logic
                    local_images.append(str(figures_dir / Path(img).name))
        # Metadata
        metadata = dict(item)
        metadata["explanation"] = explanation
        metadata["dataset"] = "chestagentbench"
        return BenchmarkDataPoint(
            id=question_id,
            text=question_with_options,
            images=local_images,
            correct_answer=correct_answer,
            metadata=metadata,
            case_id=case_id,
            category=category,
        ) 