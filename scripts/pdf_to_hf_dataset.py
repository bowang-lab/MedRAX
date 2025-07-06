#!/usr/bin/env python3
"""
PDF to HuggingFace Dataset Converter

This script converts PDF files in a directory to a format ready for HuggingFace dataset upload.
It processes PDFs, chunks them appropriately, and creates a dataset with the expected schema:
- id: unique identifier for each chunk
- title: title of the source document
- content: text content of the chunk

Usage:
    python pdf_to_hf_dataset.py --input_dir /path/to/pdfs --output_dir /path/to/output [options]
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import hashlib
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
from datasets import Dataset


class PDFToHFConverter:
    """Converter for PDF files to HuggingFace dataset format."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        """Initialize the converter with chunking configuration."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Define text splitting separators
        separators = [
            "\n\n",  # Double newlines (paragraphs)
            "\n",  # Single newlines
            ". ",  # Sentences
            "? ",  # Questions
            "! ",  # Exclamations
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Spaces
            "",  # Characters
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF file and return chunks with metadata."""
        try:
            print(f"Processing: {pdf_path}")

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            if not documents:
                print(f"Warning: No content extracted from {pdf_path}")
                return []

            # Combine all pages into one document for better chunking
            full_text = "\n\n".join([doc.page_content for doc in documents])

            # Extract title (filename without extension)
            filename = Path(pdf_path).name
            title = Path(pdf_path).stem

            # Create a single document for chunking
            combined_doc = Document(
                page_content=full_text,
                metadata={
                    "source": pdf_path,
                    "title": title,
                    "filename": filename,
                    "total_pages": len(documents),
                },
            )

            # Split into chunks
            chunks = self.text_splitter.split_documents([combined_doc])

            # Convert to HF format
            hf_chunks = []
            for i, chunk in enumerate(chunks):
                # Create unique ID using hash of content + position
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
                chunk_id = f"{Path(pdf_path).stem}_{i:04d}_{content_hash}"

                # Clean content
                content = chunk.page_content.strip()

                # Skip very short chunks
                if len(content) < 100:
                    continue

                hf_chunk = {
                    "id": chunk_id,
                    "title": title,
                    "content": content,
                    "source": pdf_path,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(content),
                }

                hf_chunks.append(hf_chunk)

            print(f"Created {len(hf_chunks)} chunks from {pdf_path}")
            return hf_chunks

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []

    def process_directory(
        self, input_dir: str, output_dir: str, output_format: str = "json"
    ) -> None:
        """Process all PDFs in a directory and save in HF format."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all PDF files
        pdf_files = list(input_path.glob("**/*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return

        print(f"Found {len(pdf_files)} PDF files to process")

        all_chunks = []

        # Process each PDF
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            chunks = self.process_pdf(str(pdf_path))
            all_chunks.extend(chunks)

        if not all_chunks:
            print("No chunks were created from any PDFs")
            return

        print(f"Total chunks created: {len(all_chunks)}")

        # Save in requested format
        if output_format.lower() == "json":
            self.save_as_json(all_chunks, output_path)
        elif output_format.lower() == "jsonl":
            self.save_as_jsonl(all_chunks, output_path)
        elif output_format.lower() == "parquet":
            self.save_as_parquet(all_chunks, output_path)
        elif output_format.lower() == "csv":
            self.save_as_csv(all_chunks, output_path)
        else:
            print(f"Unsupported format: {output_format}")
            return

        # Also save metadata
        self.save_metadata(all_chunks, output_path)

        print(f"Dataset saved to {output_path}")
        print(f"Ready for HuggingFace upload!")

    def save_as_json(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """Save chunks as JSON file."""
        output_file = output_path / "dataset.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {output_file}")

    def save_as_jsonl(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """Save chunks as JSONL file."""
        output_file = output_path / "dataset.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved JSONL: {output_file}")

    def save_as_parquet(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """Save chunks as Parquet file."""
        # Create minimal version for HF (only required fields)
        hf_data = [
            {"id": chunk["id"], "title": chunk["title"], "content": chunk["content"]}
            for chunk in chunks
        ]

        df = pd.DataFrame(hf_data)
        output_file = output_path / "dataset.parquet"
        df.to_parquet(output_file, index=False)
        print(f"Saved Parquet: {output_file}")

    def save_as_csv(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """Save chunks as CSV file."""
        df = pd.DataFrame(chunks)
        output_file = output_path / "dataset.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved CSV: {output_file}")

    def save_metadata(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """Save dataset metadata and statistics."""
        metadata = {
            "total_chunks": len(chunks),
            "total_sources": len(set(chunk["source"] for chunk in chunks)),
            "avg_chunk_size": sum(chunk["chunk_size"] for chunk in chunks) / len(chunks),
            "chunk_size_config": self.chunk_size,
            "chunk_overlap_config": self.chunk_overlap,
            "sources": list(set(chunk["source"] for chunk in chunks)),
            "titles": list(set(chunk["title"] for chunk in chunks)),
        }

        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata: {metadata_file}")


if __name__ == "__main__":
    """Main function to run the converter."""
    parser = argparse.ArgumentParser(description="Convert PDF files to HuggingFace dataset format")
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for dataset")
    parser.add_argument(
        "--format",
        "-f",
        default="parquet",
        choices=["json", "jsonl", "parquet", "csv"],
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        default=1500,
        help="Chunk size for text splitting (default: 1500)",
    )
    parser.add_argument(
        "--chunk_overlap",
        "-ol",
        type=int,
        default=300,
        help="Chunk overlap for text splitting (default: 300)",
    )

    args = parser.parse_args()

    # Create converter and process
    converter = PDFToHFConverter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    converter.process_directory(
        input_dir=args.input_dir, output_dir=args.output_dir, output_format=args.format
    )
