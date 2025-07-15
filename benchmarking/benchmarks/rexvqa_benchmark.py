"""ReXVQA benchmark implementation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datasets import load_dataset
from .base import Benchmark, BenchmarkDataPoint
import hashlib


class ReXVQABenchmark(Benchmark):
    """ReXVQA benchmark for chest radiology visual question answering.
    
    ReXVQA is a large-scale VQA dataset for chest radiology comprising approximately 
    696,000 questions paired with 160,000 chest X-rays. It tests 5 core radiological 
    reasoning skills: presence assessment, location analysis, negation detection, 
    differential diagnosis, and geometric reasoning.
    
    The dataset consists of two separate HuggingFace datasets:
    - ReXVQA: Contains questions, answers, and metadata
    - ReXGradient-160K: Contains the actual chest X-ray images
    
    Paper: https://arxiv.org/abs/2506.04353
    Dataset: https://huggingface.co/datasets/rajpurkarlab/ReXVQA
    Images: https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K
    """

    def __init__(self, data_dir: str, **kwargs):
        """Initialize ReXVQA benchmark.
        
        Args:
            data_dir (str): Directory to store/cache downloaded data
            **kwargs: Additional configuration parameters
                split (str): Dataset split to use (default: 'test')
                cache_dir (str): Directory for caching HuggingFace datasets
                trust_remote_code (bool): Whether to trust remote code (default: False)
                max_questions (int): Maximum number of questions to load (default: None, load all)
        """
        self.split = kwargs.get("split", "test")
        self.cache_dir = kwargs.get("cache_dir", None)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.max_questions = kwargs.get("max_questions", None)
        self.image_dataset = None
        self.image_mapping = {}  # Maps study_id to image data
        
        super().__init__(data_dir, **kwargs)

    def _load_data(self) -> None:
        """Load ReXVQA data from local JSON file."""
        try:
            # Construct path to the JSON file
            json_file_path = os.path.join("benchmarking", "data", "test_vqa_data.json")
            
            # Check if file exists
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"Could not find test_vqa_data.json in the expected location: {json_file_path}")
            
            print(f"Loading ReXVQA {self.split} split from local JSON file: {json_file_path}")
            
            # Load JSON file directly
            with open(json_file_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            # ReXVQA format: {question_id: {question_data}, ...}
            questions_list = []
            for question_id, question_data in questions_data.items():
                # Add the question_id to the question_data for reference
                question_data['id'] = question_id
                questions_list.append(question_data)
            
            print(f"Loaded {len(questions_list)} questions from local JSON file")
            
            # Load images dataset from ReXGradient-160K
            print("Loading ReXGradient-160K images dataset...")
            try:
                self.image_dataset = load_dataset(
                    "rajpurkarlab/ReXGradient-160K",
                    split="test",
                    cache_dir=self.cache_dir,
                    trust_remote_code=self.trust_remote_code
                )
                print(f"Loaded {len(self.image_dataset)} images from ReXGradient-160K")
                
                # Create mapping from study_id to image data
                self._create_image_mapping()
                
            except Exception as e:
                print(f"Warning: Could not load ReXGradient-160K dataset: {e}")
                print("Proceeding without images...")
                self.load_images = False
            
            self.data_points = []
            
            # Process questions (limit if max_questions is specified)
            questions_to_process = questions_list
            if self.max_questions:
                questions_to_process = questions_list[:min(self.max_questions, len(questions_list))]
            
            for i, item in enumerate(questions_to_process):
                try:
                    data_point = self._parse_rexvqa_item(item, i)
                    if data_point:
                        self.data_points.append(data_point)
                        
                except Exception as e:
                    print(f"Error loading item {i}: {e}")
                    continue
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load ReXVQA dataset: {e}")

    def _create_image_mapping(self) -> None:
        """Create mapping from study_id to image data."""
        if not self.image_dataset:
            return
            
        print("Creating image mapping...")
        
        for item in self.image_dataset:
            study_instance_uid = item.get("StudyInstanceUid", "")
            if study_instance_uid:
                # Store the image data for this study using StudyInstanceUid as key
                if study_instance_uid not in self.image_mapping:
                    self.image_mapping[study_instance_uid] = []
                self.image_mapping[study_instance_uid].append(item)
        
        print(f"Created image mapping for {len(self.image_mapping)} studies")

    def _parse_rexvqa_item(self, item: Dict[str, Any], index: int) -> Optional[BenchmarkDataPoint]:
        """Parse a ReXVQA dataset item.
        
        Args:
            item (Dict[str, Any]): Dataset item from JSON file
            index (int): Item index
            
        Returns:
            Optional[BenchmarkDataPoint]: Parsed data point
        """
        # Extract basic information
        question_id = item.get("id", f"rexvqa_{self.split}_{index}")
        question = item.get("question", "")
        
        # Handle multiple choice options
        options = item.get("options", [])
        if options:
            # Add options to the question for multiple choice format
            question_with_options = question + "\n\nOptions:\n" + "\n".join(options)
        else:
            question_with_options = question
        
        # Get correct answer 
        correct_answer = item.get("correct_answer", "")
        
        # If we have options and a letter answer, get the full text
        if options and correct_answer and len(correct_answer) == 1:
            try:
                # Find the option that starts with the correct letter
                for option in options:
                    if option.strip().startswith(f"{correct_answer}."):
                        correct_answer = option.strip()
                        break
            except:
                pass  # Keep the original letter if parsing fails
        
        if not question:
            return None
        
        # Handle images - look for ImagePath field
        images = None
        image_paths = item.get("ImagePath", [])
        study_id = item.get("study_id", "")
        study_instance_uid = item.get("StudyInstanceUid", "")
        
        if image_paths:
            # Use local image paths if available
            images = [str(Path(path)) for path in image_paths if path]
        elif study_instance_uid and study_instance_uid in self.image_mapping:
            # Use StudyInstanceUid for matching with HuggingFace images
            images = self._get_images_for_study(study_instance_uid, question_id)
        
        # Extract metadata
        metadata = {
            "dataset": "rexvqa",
            "split": self.split,
            "study_id": study_id,
            "study_instance_uid": study_instance_uid,
            "reasoning_type": item.get("task_name", ""),  # task_name maps to reasoning_type
            "category": item.get("category", ""),
            "class": item.get("class", ""),
            "subcategory": item.get("subcategory", ""),
            "patient_id": item.get("PatientID", ""),
            "patient_age": item.get("PatientAge", ""),
            "patient_sex": item.get("PatientSex", ""),
            "study_date": item.get("StudyDate", ""),
            "indication": item.get("Indication", ""),
            "findings": item.get("Findings", ""),
            "impression": item.get("Impression", ""),
            "image_modality": item.get("ImageModality", []),
            "image_view_position": item.get("ImageViewPosition", []),
            "correct_answer_explanation": item.get("correct_answer_explanation", ""),
        }
        
        # Determine category from task_name or category field
        category = item.get("task_name", item.get("category", ""))
        
        # Use study_id as case_id for grouping related questions (keep using compound study_id for grouping)
        case_id = study_id
        
        return BenchmarkDataPoint(
            id=question_id,
            text=question_with_options,
            images=images,
            correct_answer=correct_answer,
            metadata=metadata,
            case_id=case_id,
            category=category,
        )

    def _get_images_for_study(self, study_instance_uid: str, question_id: str) -> Optional[List[str]]:
        """Get images for a specific study and save them locally.
        
        Args:
            study_instance_uid (str): Study Instance UID
            question_id (str): Question ID for filename
            
        Returns:
            Optional[List[str]]: List of image paths
        """
        if study_instance_uid not in self.image_mapping:
            return None
        
        images = []
        study_images = self.image_mapping[study_instance_uid]
        
        # Create images directory if it doesn't exist
        images_dir = self.data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get every image for the study
        if not images and study_images:
            for img_data in study_images:
                image_path = self._save_image(img_data, question_id, images_dir)
                if image_path:
                    images.append(image_path)
        
        return images if images else None

    def _save_image(self, img_data: Dict[str, Any], question_id: str, images_dir) -> Optional[str]:
        """Save image data to local file.
        
        Args:
            img_data (Dict[str, Any]): Image data from dataset
            question_id (str): Question ID for filename
            images_dir: Directory to save images
            
        Returns:
            Optional[str]: Path to saved image
        """
        try:
            # Get the image from the dataset item
            image = img_data.get("image")
            if image is None:
                return None
            
            # Generate filename using StudyInstanceUid
            study_instance_uid = img_data.get("StudyInstanceUid", "")
            filename_hash = hashlib.md5(f"{question_id}_{study_instance_uid}".encode()).hexdigest()[:8]
            image_filename = f"{question_id}_{filename_hash}.png"
            image_path = images_dir / image_filename
            
            # Save image if it doesn't exist
            if not image_path.exists():
                image.save(str(image_path))
                
            return str(image_path)
            
        except Exception as e:
            print(f"Error saving image for question {question_id}: {e}")
            return None
