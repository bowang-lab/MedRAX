#!/usr/bin/env python3
"""
Utility script to download and extract ReXGradient-160K images.

This script helps users download the actual PNG images from the ReXGradient-160K dataset,
which are stored as part files on HuggingFace and need to be concatenated and extracted.

Usage:
    python download_rexgradient_images.py --output_dir /path/to/images
"""

import argparse
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import requests
from tqdm import tqdm


def download_file(url, output_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download ReXGradient-160K images")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save extracted images"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="rajpurkarlab/ReXGradient-160K",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading and only extract if files exist"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Check if we need to accept the license first
    print("Note: You may need to accept the dataset license on HuggingFace first:")
    print(f"Visit: https://huggingface.co/datasets/{args.repo_id}")
    print("Click 'Access repository' and accept the license agreement.")
    print()
    
    try:
        # List files in the repository
        print("Listing files in repository...")
        files = list_repo_files(args.repo_id, repo_type='dataset')
        part_files = [f for f in files if f.startswith("deid_png.part")]
        
        if not part_files:
            print("No part files found. The images might be in a different format.")
            print("Available files:")
            for f in files:
                print(f"  - {f}")
            return
        
        print(f"Found {len(part_files)} part files:")
        for f in part_files:
            print(f"  - {f}")
        
        # Download part files
        if not args.skip_download:
            print("\nDownloading part files...")
            for part_file in part_files:
                output_path = output_dir / part_file
                if output_path.exists():
                    print(f"Skipping {part_file} (already exists)")
                    continue
                
                print(f"Downloading {part_file}...")
                try:
                    hf_hub_download(
                        repo_id=args.repo_id,
                        filename=part_file,
                        local_dir=output_dir,
                        local_dir_use_symlinks=False,
                        repo_type='dataset'
                    )
                except Exception as e:
                    print(f"Error downloading {part_file}: {e}")
                    print("You may need to accept the license agreement on HuggingFace.")
                    return
        
        # Concatenate part files
        tar_path = output_dir / "deid_png.tar"
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
        else:
            print(f"Tar file already exists: {tar_path}")
        
        # Extract tar file
        if tar_path.exists():
            print("\nExtracting images...")
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Check if already extracted
            if any(images_dir.glob("*.png")):
                print("Images already extracted.")
            else:
                try:
                    subprocess.run([
                        "tar", "-xf", str(tar_path), 
                        "-C", str(images_dir)
                    ], check=True)
                    print("Extraction completed!")
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting tar file: {e}")
                    return
                except FileNotFoundError:
                    print("Error: 'tar' command not found. Please install tar or extract manually.")
                    return
            
            # Count extracted images
            png_files = list(images_dir.glob("*.png"))
            print(f"Extracted {len(png_files)} PNG images to {images_dir}")
            
            # Show some example filenames
            if png_files:
                print("\nExample image filenames:")
                for f in png_files[:5]:
                    print(f"  - {f.name}")
                if len(png_files) > 5:
                    print(f"  ... and {len(png_files) - 5} more")
        
        print(f"\nSetup complete! Use this directory as images_dir in ReXVQABenchmark:")
        print(f"images_dir='{images_dir}'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nManual setup instructions:")
        print("1. Visit https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K")
        print("2. Accept the license agreement")
        print("3. Download the deid_png.part* files")
        print("4. Concatenate: cat deid_png.part* > deid_png.tar")
        print("5. Extract: tar -xf deid_png.tar")
        print("6. Use the extracted directory as images_dir")


if __name__ == "__main__":
    main() 