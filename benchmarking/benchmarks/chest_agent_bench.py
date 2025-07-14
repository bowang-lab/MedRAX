"""ChestAgentBench benchmark implementation."""

import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Benchmark, BenchmarkDataPoint


class ChestAgentBench(Benchmark):
    """ChestAgentBench benchmark for complex medical reasoning tasks."""

    def __init__(self, data_dir: str, **kwargs):
        """Initialize ChestAgentBench.
        
        Args:
            data_dir (str): Directory containing benchmark data
            **kwargs: Additional configuration parameters
        """
        # Expected structure:
        # data_dir/
        #   eurorad_metadata.json  # Case metadata
        #   questions/
        #     case_id1/
        #       case_id1_question1.json
        #       case_id1_question2.json
        #     case_id2/
        #       ...
        #   figures/
        #     case_id1/
        #       figure1.jpg
        #       figure2.jpg
        #     case_id2/
        #       ...
        
        self.metadata_file = kwargs.get("metadata_file", "eurorad_metadata.json")
        self.questions_dir = kwargs.get("questions_dir", "questions")
        self.figures_dir = kwargs.get("figures_dir", "figures")
        
        super().__init__(data_dir, **kwargs)

    def _load_data(self) -> None:
        """Load ChestAgentBench data."""
        # Load case metadata
        metadata_path = self.data_dir / self.metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            case_metadata = json.load(f)
        
        # Load questions for each case
        questions_path = self.data_dir / self.questions_dir
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions directory not found: {questions_path}")
        
        figures_path = self.data_dir / self.figures_dir
        
        self.data_points = []
        
        for case_id, case_details in case_metadata.items():
            # Find all question files for this case
            case_questions_dir = questions_path / case_id
            if not case_questions_dir.exists():
                continue
                
            question_files = glob.glob(str(case_questions_dir / f"{case_id}_*.json"))
            
            for question_file in question_files:
                try:
                    with open(question_file, 'r') as f:
                        question_data = json.load(f)
                    
                    question_id = Path(question_file).stem
                    
                    # Parse figure information
                    images = []
                    if question_data.get("figures"):
                        required_figures = self._parse_figures(question_data["figures"])
                        
                        # Find actual image files
                        case_figures_dir = figures_path / case_id
                        if case_figures_dir.exists():
                            for figure_id in required_figures:
                                # Look for the figure file
                                figure_files = glob.glob(str(case_figures_dir / f"{figure_id}.*"))
                                if figure_files:
                                    images.append(figure_files[0])  # Take the first match
                    
                    # Extract categories from metadata
                    categories = []
                    if question_data.get("metadata", {}).get("categories"):
                        categories = question_data["metadata"]["categories"]
                    
                    category = categories[0] if categories else None
                    
                    # Create data point
                    data_point = BenchmarkDataPoint(
                        id=question_id,
                        text=question_data["question"],
                        images=images if images else None,
                        correct_answer=question_data.get("answer", [None])[0],
                        metadata={
                            "case_details": case_details,
                            "question_metadata": question_data.get("metadata", {}),
                            "explanation": question_data.get("explanation", ""),
                            "categories": categories,
                            "figures": question_data.get("figures", []),
                        },
                        case_id=case_id,
                        category=category,
                    )
                    
                    self.data_points.append(data_point)
                    
                except Exception as e:
                    print(f"Error loading question {question_file}: {e}")
                    continue

    def _parse_figures(self, figures_data: Any) -> List[str]:
        """Parse figure information from question data.
        
        Args:
            figures_data: Figure information from question JSON
            
        Returns:
            List[str]: List of figure IDs
        """
        if isinstance(figures_data, str):
            try:
                # Try to parse as JSON
                figures_list = json.loads(figures_data)
                return figures_list if isinstance(figures_list, list) else [figures_data]
            except json.JSONDecodeError:
                return [figures_data]
        elif isinstance(figures_data, list):
            return figures_data
        else:
            return [str(figures_data)]

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

    def get_multiple_choice_options(self, data_point: BenchmarkDataPoint) -> List[str]:
        """Get multiple choice options for a data point.
        
        Args:
            data_point (BenchmarkDataPoint): The data point
            
        Returns:
            List[str]: List of multiple choice options (A, B, C, D, E, F)
        """
        # ChestAgentBench uses A-F multiple choice
        return ["A", "B", "C", "D", "E", "F"]

    def format_question_with_choices(self, data_point: BenchmarkDataPoint) -> str:
        """Format question text with multiple choice options.
        
        Args:
            data_point (BenchmarkDataPoint): The data point
            
        Returns:
            str: Formatted question with choices
        """
        question = data_point.text
        
        # Add instruction for multiple choice format
        question += "\n\nPlease provide your answer as a single letter (A, B, C, D, E, or F)."
        
        return question

    def get_category_mapping(self) -> Dict[str, str]:
        """Get mapping of category names to descriptions.
        
        Returns:
            Dict[str, str]: Mapping of category names to descriptions
        """
        return {
            "detection": "Identify and locate specific findings in the chest X-ray",
            "classification": "Determine whether specific findings are present or absent",
            "enumeration": "Count the number of target findings in the chest X-ray",
            "localization": "Locate a given finding in the chest X-ray",
            "comparison": "Compare the size or position of a specific finding",
            "relationship": "Determine the relationship between two or more findings",
            "diagnosis": "Make a diagnosis or determine a treatment plan",
            "characterization": "Describe specific attributes of findings",
            "reasoning": "Explain the medical rationale behind findings and conclusions",
        } 