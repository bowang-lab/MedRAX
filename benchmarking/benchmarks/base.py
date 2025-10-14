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

        self.max_questions = kwargs.get("max_questions", None)
        if self.max_questions:
            self.data_points = self.data_points[:self.max_questions]
            print(f"Randomly sampled {self.max_questions} questions from {self.__class__.__name__}")
        else:
            print(f"Loaded all {len(self.data_points)} questions from {self.__class__.__name__}")

    @abstractmethod
    def _load_data(self) -> None:
        """Load benchmark data from the data directory."""
        pass

    def _shuffle_data(self, random_seed: Optional[int]=42) -> None:
        """Shuffle the data points if a random seed is provided. If no random seed is provided, use 42 as default.
        
        This method is called automatically after data loading to ensure
        reproducible benchmark runs when a random seed is specified.
        """
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
