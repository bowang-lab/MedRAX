import json
from pathlib import Path
from typing import Dict, Optional, Any
from .base import Benchmark, BenchmarkDataPoint

class ChestAgentBenchBenchmark(Benchmark):
    """ChestAgentBench benchmark for complex CXR interpretation and reasoning.
    
    Loads the dataset from a local metadata.jsonl file and parses each entry into a BenchmarkDataPoint.
    """
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, **kwargs)

    def _load_data(self) -> None:
        # Check if metadata.jsonl exists
        metadata_path = Path(self.data_dir) / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Could not find metadata.jsonl in {self.data_dir}")
        print(f"Loading ChestAgentBench from local file: {metadata_path}")

        # Load metadata.jsonl
        with open(metadata_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    data_point = self._parse_item(item, i)
                    if data_point:
                        self.data_points.append(data_point)
                except Exception as e:
                    print(f"Error loading item {i}: {e}")
                    continue
        

    def _parse_item(self, item: Dict[str, Any], index: int) -> Optional[BenchmarkDataPoint]:
        # Extract required fields
        question_id = item.get("full_question_id")
        question = item.get("question", "")
        correct_answer = item.get("answer", "")

        # Map image paths to local figures directory
        images = item.get("images", [])
        local_images = None
        if images:
            local_images = []
            for img in images:
                full_path = Path(self.data_dir) / img
                local_images.append(str(full_path))
        
        # Extract metadata
        metadata = dict(item)
        metadata["dataset"] = "chestagentbench"

        # Return data point
        return BenchmarkDataPoint(
            id=question_id,
            text=question,
            images=local_images,
            correct_answer=correct_answer,
            metadata=metadata,
        ) 