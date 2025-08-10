"""Base class for benchmarks."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path
import random


@dataclass
class BenchmarkDataPoint:
    """A single data point from a benchmark."""
    id: str
    text: str  # The question/prompt
    images: Optional[List[str]] = None  # List of image paths
    correct_answer: Optional[str] = None  # Ground truth answer
    case_id: Optional[str] = None  # For grouping related questions
    category: Optional[str] = None  # Type of question/task
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class Benchmark(ABC):
    """Abstract base class for benchmarks.
    
    This class defines the interface for all benchmarks, standardizing
    how data is loaded and accessed across different benchmark datasets.
    """

    def __init__(self, data_dir: str, **kwargs):
        """Initialize the benchmark.
        
        Args:
            data_dir (str): Directory containing benchmark data
            **kwargs: Additional configuration parameters
                random_seed (int): Random seed for shuffling data (default: None, no shuffling)
        """
        self.data_dir = Path(data_dir)
        self.config = kwargs
        self.data_points = []
        self._load_data()
        self._shuffle_data()

    @abstractmethod
    def _load_data(self) -> None:
        """Load benchmark data from the data directory."""
        pass

    def _shuffle_data(self) -> None:
        """Shuffle the data points if a random seed is provided.
        
        This method is called automatically after data loading to ensure
        reproducible benchmark runs when a random seed is specified.
        """
        random_seed = self.config.get("random_seed", None)
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(self.data_points)
            print(f"Shuffled {len(self.data_points)} data points with seed {random_seed}")

    def get_data_point(self, index: int) -> BenchmarkDataPoint:
        """Get a specific data point by index.
        
        Args:
            index (int): Index of the data point to retrieve
            
        Returns:
            BenchmarkDataPoint: The data point at the given index
        """
        if index < 0 or index >= len(self.data_points):
            raise IndexError(f"Index {index} out of range for {len(self.data_points)} data points")
        
        return self.data_points[index]

    def get_subset(self, indices: List[int]) -> List[BenchmarkDataPoint]:
        """Get a subset of data points by indices.
        
        Args:
            indices (List[int]): List of indices to retrieve
            
        Returns:
            List[BenchmarkDataPoint]: List of data points at the given indices
        """
        return [self.get_data_point(i) for i in indices]

    def get_by_category(self, category: str) -> List[BenchmarkDataPoint]:
        """Get all data points of a specific category.
        
        Args:
            category (str): Category to filter by
            
        Returns:
            List[BenchmarkDataPoint]: List of data points in the category
        """
        return [dp for dp in self if dp.category == category]

    def get_by_case_id(self, case_id: str) -> List[BenchmarkDataPoint]:
        """Get all data points for a specific case.
        
        Args:
            case_id (str): Case ID to filter by
            
        Returns:
            List[BenchmarkDataPoint]: List of data points for the case
        """
        return [dp for dp in self if dp.case_id == case_id]

    def __str__(self) -> str:
        """String representation of the benchmark."""
        return f"{self.__class__.__name__}(data_dir={self.data_dir}, size={len(self)})" 

    def __len__(self) -> int:
        """Return the number of data points in the benchmark."""
        return len(self.data_points)

    def __iter__(self) -> Iterator[BenchmarkDataPoint]:
        """Iterate over all data points in the benchmark."""
        for i in range(len(self)):
            yield self.get_data_point(i)

    def get_categories(self) -> List[str]:
        """Get all unique categories in the benchmark.
        
        Returns:
            List[str]: List of unique categories
        """
        categories = set()
        for dp in self:
            if dp.category:
                categories.add(dp.category)
        return sorted(list(categories))

    def get_case_ids(self) -> List[str]:
        """Get all unique case IDs in the benchmark.
        
        Returns:
            List[str]: List of unique case IDs
        """
        case_ids = set()
        for dp in self:
            if dp.case_id:
                case_ids.add(dp.case_id)
        return sorted(list(case_ids))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the benchmark.
        
        Returns:
            Dict[str, Any]: Dictionary containing benchmark statistics
        """
        stats = {
            "total_questions": len(self),
            "total_cases": len(self.get_case_ids()),
            "categories": self.get_categories(),
            "category_counts": {},
            "has_images": False,
            "num_images": 0,
        }
        
        for dp in self:
            # Category counts
            if dp.category:
                stats["category_counts"][dp.category] = stats["category_counts"].get(dp.category, 0) + 1
            
            # Image statistics
            if dp.images:
                stats["has_images"] = True
                stats["num_images"] += len(dp.images)
        return stats

    def validate_images(self) -> Tuple[List[str], List[str]]:
        """Validate that all image paths exist.
        
        Returns:
            Tuple[List[str], List[str]]: Tuple of (valid_image_paths, invalid_image_paths)
        """
        valid_images = []
        invalid_images = []
        
        for dp in self:
            if dp.images:
                for image_path in dp.images:
                    if Path(image_path).exists():
                        valid_images.append(image_path)
                    else:
                        invalid_images.append(image_path)
        
        return valid_images, invalid_images
