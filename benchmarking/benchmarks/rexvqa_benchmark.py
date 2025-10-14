"""ReXVQA benchmark implementation."""

import json
import os
from typing import Dict, Optional, Any
from .base import Benchmark, BenchmarkDataPoint
from pathlib import Path
import tarfile
import zstandard as zstd
from huggingface_hub import hf_hub_download, list_repo_files
import os


def get_hf_token():
    """Get Hugging Face token from cache."""
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()
    return None


class ReXVQABenchmark(Benchmark):
    """ReXVQA benchmark for chest radiology visual question answering.
    
    ReXVQA is a large-scale VQA dataset for chest radiology comprising approximately 
    696,000 questions paired with 160,000 chest X-rays. It tests 5 core radiological 
    reasoning skills: presence assessment, location analysis, negation detection, 
    differential diagnosis, and geometric reasoning.
    
    The dataset consists of two separate HuggingFace datasets:
    - ReXVQA: Contains questions, answers, and metadata
    - ReXGradient-160K: Contains metadata only (images are in separate part files)
    
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
                trust_remote_code (bool): Whether to trust remote code (default: False)
                max_questions (int): Maximum number of questions to load (default: None, load all)
                images_dir (str): Directory containing extracted PNG images (default: None)
        """
        super().__init__(data_dir, **kwargs)

        self.split = kwargs.get("split", "test")
        self.images_dir = f"{data_dir}/images/deid_png"

    def _load_data(self) -> None:
        """Load ReXVQA data from HuggingFace."""
        try:
            # Download images and test_vqa_data.json locally if missing
            self.download_test_vqa_data_json(self.data_dir)
            self.download_rexgradient_images(self.data_dir, test_only=True)
            
            # Load JSON file
            json_file_path = os.path.join(self.data_dir, "metadata", "test_vqa_data.json")
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"Could not find test_vqa_data.json in the expected location: {json_file_path}")
            print(f"Loading ReXVQA {self.split} split from local JSON file: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            # ReXVQA format: {question_id: {question_data}, ...}
            questions_list = []
            for question_id, question_data in questions_data.items():
                # Add the question_id to the question_data for reference
                question_data['id'] = question_id
                questions_list.append(question_data)
            print(f"Loaded {len(questions_list)} questions from local JSON file")
    
            # Process questions
            for i, item in enumerate(questions_list):
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
            item (Dict[str, Any]): Dataset item from JSON file
            index (int): Item index
            
        Returns:
            Optional[BenchmarkDataPoint]: Parsed data point
        """
        # Extract question ID
        question_id = item.get("id", f"rexvqa_{self.split}_{index}")

        # Extract question and options
        question = item.get("question", "")
        options = item.get("options", [])
        question_with_options = question + "\n\nOptions:\n" + "\n".join(options)
        
        # Extract correct answer
        correct_answer = item.get("correct_answer", "")
        
        # Extract images
        images = None
        if self.images_dir and "ImagePath" in item and item["ImagePath"]:
            images = []
            for rel_path in item["ImagePath"]:
                norm_rel_path = rel_path.lstrip("./")
                full_path = str(Path(self.images_dir).parent / norm_rel_path)
                images.append(full_path)
        
        # Extract metadata
        metadata = {
            "dataset": "rexvqa",
            "split": self.split,
            "study_id": item.get("study_id", ""),
            "study_instance_uid": item.get("StudyInstanceUid", ""),
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

        # Return data point
        return BenchmarkDataPoint(
            id=question_id,
            text=question_with_options,
            images=images,
            correct_answer=correct_answer,
            metadata=metadata
        )

    @staticmethod
    def download_rexgradient_images(output_dir: str = "benchmarking/data/rexvqa", repo_id: str = "rajpurkarlab/ReXGradient-160K", test_only: bool = True):
        """Download and extract ReXGradient-160K images if not already present.
        
        Args:
            output_dir: Directory to store downloaded and extracted images
            repo_id: HuggingFace repository ID for the dataset
            test_only: If True, only extract images from the test split (default: True)
        """
        output_dir = Path(output_dir)
        tar_path = output_dir / "deid_png.tar"
        images_dir = output_dir / "images"

        # Check if images already exist
        if images_dir.exists() and any(images_dir.rglob("*.png")):
            print(f"Images already exist in {images_dir}, skipping download.")
            return
            
        # Load test split metadata if test_only is True
        test_image_paths = set()
        if test_only:
            print("Loading test split metadata to identify test images...")
            try:
                # Load the test metadata to get image paths
                test_metadata_path = output_dir / "metadata" / "test_vqa_data.json"
                if test_metadata_path.exists():
                    with open(test_metadata_path, 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                    
                    # Extract all image paths from test data
                    for item in test_data.values():
                        if "ImagePath" in item and item["ImagePath"]:
                            for rel_path in item["ImagePath"]:
                                # Normalize path to match tar file structure
                                norm_path = rel_path.lstrip("./")
                                test_image_paths.add(norm_path)
                    
                    print(f"Found {len(test_image_paths)} test images to extract")
                else:
                    print("Warning: test_vqa_data.json not found, will extract all images")
                    test_only = False
            except Exception as e:
                print(f"Warning: Could not load test metadata: {e}, will extract all images")
                test_only = False
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        try:
            print("Listing files in repository...")
            files = list_repo_files(repo_id, repo_type='dataset', token=get_hf_token())
            part_files = [f for f in files if f.startswith("deid_png.part")]
            if not part_files:
                print("No part files found. The images might be in a different format.")
                return
            print(f"Found {len(part_files)} part files.")
            # Download part files
            for part_file in part_files:
                output_path = output_dir / part_file
                if output_path.exists():
                    print(f"Skipping {part_file} (already exists)")
                    continue
                print(f"Downloading {part_file}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=part_file,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                    repo_type='dataset',
                    token=get_hf_token()
                )
            # Concatenate part files
            if not tar_path.exists():
                print("\nConcatenating part files...")
                with open(tar_path, 'wb') as tar_file:
                    for part_file in sorted(part_files):
                        part_path = output_dir / part_file
                        if part_path.exists():
                            print(f"Adding {part_file}...")
                            with open(part_path, 'rb') as f:
                                tar_file.write(f.read())
                        else:
                            print(f"Warning: {part_file} not found, skipping...")
                
                # Clean up part files after successful concatenation
                print("Cleaning up part files...")
                for part_file in part_files:
                    part_path = output_dir / part_file
                    if part_path.exists():
                        try:
                            part_path.unlink()
                            print(f"Deleted {part_file}")
                        except Exception as e:
                            print(f"Could not delete {part_file}: {e}")
            else:
                print(f"Tar file already exists: {tar_path}")
            # Extract tar file
            if tar_path.exists():
                print("\nExtracting images...")
                images_dir.mkdir(exist_ok=True)
                if any(images_dir.rglob("*.png")):
                    print("Images already extracted.")
                else:
                    try:
                        # Stream extract with filtering for test-only images (no seeking)
                        print("Stream extracting zstd-compressed tar file with filtering (streaming mode)...")

                        # Create a decompressor
                        dctx = zstd.ZstdDecompressor()

                        # Stream extract with filtering
                        extracted_count = 0
                        total_png_members = 0

                        with open(tar_path, 'rb') as compressed_file:
                            with dctx.stream_reader(compressed_file) as decompressed_stream:
                                # Use streaming tar mode to avoid seeks
                                with tarfile.open(fileobj=decompressed_stream, mode='r|') as tar:
                                    for member in tar:
                                        # Only consider PNG files
                                        if not member.isfile() or not member.name.endswith('.png'):
                                            continue
                                        total_png_members += 1

                                        # Normalize name to match entries gathered from JSON
                                        normalized_name = member.name.lstrip('./')

                                        # Decide whether to extract this file
                                        should_extract = True
                                        if test_only:
                                            should_extract = normalized_name in test_image_paths

                                        if not should_extract:
                                            # Must still advance the stream for this member
                                            tar.members = []  # no-op in stream mode; ensure we don't hold refs
                                            continue

                                        # Ensure parent directories exist and write file by streaming
                                        target_path = Path(images_dir) / normalized_name
                                        target_path.parent.mkdir(parents=True, exist_ok=True)

                                        extracted_file_obj = tar.extractfile(member)
                                        if extracted_file_obj is None:
                                            continue
                                        with open(target_path, 'wb') as out_f:
                                            while True:
                                                chunk = extracted_file_obj.read(1024 * 1024)
                                                if not chunk:
                                                    break
                                                out_f.write(chunk)

                                        extracted_count += 1
                                        if extracted_count % 100 == 0:
                                            print(f"Extracted {extracted_count} test images...")

                        print(f"Extraction completed! Extracted {extracted_count} matching PNGs out of {total_png_members} PNG members in the archive")
                        
                        # Clean up compressed tar file after successful extraction
                        print("Cleaning up compressed tar file...")
                        try:
                            tar_path.unlink()
                            print(f"Deleted {tar_path}")
                        except Exception as e:
                            print(f"Could not delete {tar_path}: {e}")
                    except Exception as e:
                        print(f"Error extracting tar file: {e}")
                        return
                png_files = list(images_dir.rglob("*.png"))
                print(f"Extracted {len(png_files)} PNG images to {images_dir}")

        except Exception as e:
            print(f"Error: {e}")

    @staticmethod
    def download_test_vqa_data_json(output_dir: str = "benchmarking/data/rexvqa", repo_id: str = "rajpurkarlab/ReXVQA"):
        """Download test_vqa_data.json from the ReXVQA HuggingFace repo if not already present."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "metadata" / "test_vqa_data.json"
        if json_path.exists():
            print(f"test_vqa_data.json already exists at {json_path}, skipping download.")
            return
        print(f"Downloading test_vqa_data.json to {json_path}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="metadata/test_vqa_data.json",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                repo_type='dataset',
                token=get_hf_token()
            )
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading test_vqa_data.json: {e}")
            print("You may need to accept the license agreement on HuggingFace.")