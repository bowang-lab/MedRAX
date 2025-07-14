"""ReXVQA benchmark implementation."""

from typing import Dict, List, Optional, Any
from datasets import load_dataset
from .base import Benchmark, BenchmarkDataPoint


class ReXVQABenchmark(Benchmark):
    """ReXVQA benchmark for chest radiology visual question answering.
    
    ReXVQA is a large-scale VQA dataset for chest radiology comprising approximately 
    696,000 questions paired with 160,000 chest X-rays. It tests 5 core radiological 
    reasoning skills: presence assessment, location analysis, negation detection, 
    differential diagnosis, and geometric reasoning.
    
    Paper: https://arxiv.org/abs/2506.04353
    Dataset: https://huggingface.co/datasets/rajpurkarlab/ReXVQA
    """

    def __init__(self, data_dir: str, **kwargs):
        """Initialize ReXVQA benchmark.
        
        Args:
            data_dir (str): Directory to store/cache downloaded data
            **kwargs: Additional configuration parameters
                split (str): Dataset split to use ('validation' or 'test', default: 'validation')
                cache_dir (str): Directory for caching HuggingFace datasets
                trust_remote_code (bool): Whether to trust remote code (default: False)
        """
        self.split = kwargs.get("split", "validation")
        self.cache_dir = kwargs.get("cache_dir", None)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        
        super().__init__(data_dir, **kwargs)

    def _load_data(self) -> None:
        """Load ReXVQA data from HuggingFace."""
        try:
            # Load dataset from HuggingFace
            print(f"Loading ReXVQA {self.split} split from HuggingFace...")
            
            dataset = load_dataset(
                "rajpurkarlab/ReXVQA",
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code
            )
            
            print(f"Loaded {len(dataset)} examples from ReXVQA {self.split} split")
            
            self.data_points = []
            
            for i, item in enumerate(dataset):
                try:
                    data_point = self._parse_rexvqa_item(item, i)
                    if data_point:
                        self.data_points.append(data_point)
                        
                except Exception as e:
                    print(f"Error loading item {i}: {e}")
                    continue
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load ReXVQA dataset: {e}")

    def _parse_rexvqa_item(self, item: Dict[str, Any], index: int) -> Optional[BenchmarkDataPoint]:
        """Parse a ReXVQA dataset item.
        
        Args:
            item (Dict[str, Any]): Dataset item from HuggingFace
            index (int): Item index
            
        Returns:
            Optional[BenchmarkDataPoint]: Parsed data point
        """
        # Extract basic information
        question_id = item.get("id", f"rexvqa_{self.split}_{index}")
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        if not question:
            return None
        
        # Handle image
        images = None
        if "image" in item and item["image"] is not None:
            # Save image to local cache directory
            image_filename = f"{question_id}.png"
            image_path = self.data_dir / "images" / image_filename
            
            # Create images directory if it doesn't exist
            image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image if it doesn't exist
            if not image_path.exists():
                try:
                    item["image"].save(str(image_path))
                except Exception as e:
                    print(f"Error saving image for {question_id}: {e}")
                    return None
                    
            images = [str(image_path)]
        
        # Extract metadata
        metadata = {
            "dataset": "rexvqa",
            "split": self.split,
            "study_id": item.get("study_id", ""),
            "image_id": item.get("image_id", ""),
            "reasoning_type": item.get("reasoning_type", ""),
            "anatomical_location": item.get("anatomical_location", ""),
            "pathology": item.get("pathology", ""),
        }
        
        # Determine category from reasoning type
        category = item.get("reasoning_type", "")
        
        # Use study_id as case_id for grouping related questions
        case_id = item.get("study_id", "")
        
        return BenchmarkDataPoint(
            id=question_id,
            text=question,
            images=images,
            correct_answer=answer,
            metadata=metadata,
            case_id=case_id,
            category=category,
        )

    def get_pathologies(self) -> List[str]:
        """Get all unique pathologies in the dataset.
        
        Returns:
            List[str]: List of unique pathologies
        """
        pathologies = set()
        for dp in self:
            pathology = dp.metadata.get("pathology", "")
            if pathology:
                pathologies.add(pathology)
        return sorted(list(pathologies))

    def get_by_pathology(self, pathology: str) -> List[BenchmarkDataPoint]:
        """Get all data points about a specific pathology.
        
        Args:
            pathology (str): Pathology to filter by
            
        Returns:
            List[BenchmarkDataPoint]: List of data points about the pathology
        """
        return [dp for dp in self if dp.metadata.get("pathology", "") == pathology]

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the ReXVQA dataset.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        return {
            "name": "ReXVQA",
            "description": "Large-scale Visual Question Answering Benchmark for Chest Radiology",
            "split": self.split,
            "size": len(self.data_points),
            "reasoning_types": self.get_reasoning_types(),
            "pathologies": self.get_pathologies(),
            "categories": self.get_categories(),
            "paper": "https://arxiv.org/abs/2506.04353",
            "dataset_url": "https://huggingface.co/datasets/rajpurkarlab/ReXVQA",
        } 