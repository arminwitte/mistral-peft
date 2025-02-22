import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
import torch
from datasets import Dataset
from pypdf import PdfReader
from tqdm import tqdm

################### TEXT EXTRACTOR ###################


class TextExtractor:
    def __init__(self, output_file: Union[str, Path], max_workers: int = 4):
        """Initialize text extractor for LLM finetuning

        Args:
            output_file: Path to save the final JSON output
            max_workers: Maximum number of concurrent workers for processing
        """
        self.output_file = Path(output_file)
        self.max_workers = max_workers
        self.http_client = httpx.Client(timeout=30.0)
        self.results = []

    def read_pdf(self, file_path: Union[str, Path]) -> str:
        """Read text from PDF file using pypdf.

        Uses layout extraction mode for better text formatting and handles
        rotated text.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        text_parts = []
        reader = PdfReader(path)

        for page in reader.pages:
            # Extract text using layout mode for better formatting
            # Include rotated text and adjust spacing for better results
            page_text = page.extract_text(
                # extraction_mode="layout",
                # layout_mode_strip_rotated=False,
                # layout_mode_scale_weight=1.25,
                # layout_mode_space_vertically=False
            )
            if page_text:
                text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def download_pdf(self, url: str) -> Path:
        """Download PDF from URL to temporary file."""
        response = self.http_client.get(url)
        response.raise_for_status()

        # Create temp file with .pdf extension
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(response.content)
        tmp.close()
        return Path(tmp.name)

    def clean_text(self, text: str) -> str:
        """Clean extracted text for LLM finetuning."""
        # Add this before other cleaning steps
        # Look for common patterns of glued words like capitalLetters
        text = re.sub(
            r"(?<=[a-z])(?=[A-Z])", " ", text
        )  # Split "wordWord" into "word Word"
        text = re.sub(
            r"(?<=[A-Za-z])(?=\d)", " ", text
        )  # Split "word123" into "word 123"
        text = re.sub(
            r"(?<=\d)(?=[A-Za-z])", " ", text
        )  # Split "123word" into "123 word"

        # Remove headers and footers
        text = re.sub(r"\f", "\n", text)
        text = re.sub(
            r"^\s*\d+\s*$", "", text, flags=re.MULTILINE
        )  # Standalone page numbers
        text = re.sub(r"(?i)(page|seite)\s*\d+\s*(?:of|von)\s*\d+", "", text)

        # Clean scientific formatting
        text = re.sub(r"(?<=\d)\.(?=\d)", ".0", text)  # Add leading 0 after decimal
        text = re.sub(
            r"×10\^(-?\d+)", lambda m: f"e{m.group(1)}", text
        )  # Fix scientific notation

        # Clean references and citations
        text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)  # Remove citation brackets
        text = re.sub(r"\(\d{4}\)", "", text)  # Remove year citations
        text = re.sub(r"et al\.", "et al", text)  # Standardize et al

        # Fix paragraph breaks
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Standardize paragraph breaks
        text = re.sub(
            r"([.!?])\n(?=[A-Z])", r"\1\n\n", text
        )  # Add breaks after sentences

        # Clean whitespace
        text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
        text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)  # Trim lines

        # Remove redundant whitespace around punctuation
        text = re.sub(r"\s+([.,;?!])", r"\1", text)
        text = re.sub(r'"\s+([^"]+)\s+"', r'"\1"', text)

        # Remove successive punctuation marks, keeping only the first one
        text = re.sub(r"([.,;?!])[.,;?!]+", r"\1", text)

        return text.strip()

    def process_single_document(
        self, source: Union[str, Path], is_url: bool = False
    ) -> Dict:
        """Process a single document from file or URL."""
        try:
            if is_url:
                tmp_path = self.download_pdf(source)
                raw_text = self.read_pdf(tmp_path)
                os.unlink(tmp_path)  # Clean up temp file
            else:
                raw_text = self.read_pdf(source)

            clean_text = self.clean_text(raw_text)

            return {
                "metadata": {
                    "source": str(source),
                    "text_length": len(clean_text),
                    "is_url": is_url,
                },
                "text": clean_text,
            }

        except Exception as e:
            return {
                "metadata": {"source": str(source), "error": str(e), "is_url": is_url},
                "text": "",
            }

    def process_documents(
        self, sources: List[Union[str, Path]], url_list: bool = False
    ) -> None:
        """Process multiple documents in parallel and store results.

        Args:
            sources: List of file paths or URLs
            url_list: If True, treat sources as URLs
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_single_document, source, url_list)
                for source in sources
            ]

            for future in tqdm(
                futures, total=len(sources), desc="Processing documents"
            ):
                self.results.append(future.result())

    def __enter__(self):
        """Context manager entry."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit. Saves results and cleans up."""
        try:
            if self.results:
                with self.output_file.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "documents": self.results,
                            "total_documents": len(self.results),
                            "successful_documents": sum(
                                1
                                for doc in self.results
                                if not doc["metadata"].get("error")
                            ),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
        finally:
            self.http_client.close()


class CLMPreprocessor:
    def __init__(self, json_files, tokenizer, nlp=None):
        """
        Initializes the CLAPreprocessor.

        Args:
            json_files (list or str): A list of paths to JSON files or a single path.
            tokenizer: The Hugging Face tokenizer to use.
            nlp: An optional spaCy language model. If None, language detection will be attempted.
        """

        if isinstance(json_files, str):
            json_files = [json_files]

        self.json_files = json_files
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.nlp = nlp
        self.dataset = None

    def _read_json(self, file_path):
        """Reads a single JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)["documents"]
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON format in: {file_path}")

    def _process_data(self, chunk_size=4096):  # Add chunk_size parameter
        all_sentences = []
        for file_path in self.json_files:
            data = self._read_json(file_path)
            all_sentences += [d["text"] for d in data]

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for sentence in all_sentences:
            tokenized_sentence = self.tokenizer(
                sentence, truncation=False, padding=False, return_tensors="pt"
            )  # No padding/truncation here
            input_ids = tokenized_sentence["input_ids"]
            attention_mask = tokenized_sentence["attention_mask"]

            # Create chunks
            for i in range(
                0, input_ids.size(1), chunk_size - 1
            ):  # Changed range to chunk_size-1 as we want to shift by one
                chunk_input_ids = input_ids[:, i : i + chunk_size]
                chunk_attention_mask = attention_mask[:, i : i + chunk_size]

                # Ensure chunk is of correct length by padding if necessary
                padding_length = chunk_size - chunk_input_ids.size(1)
                if padding_length > 0:
                    chunk_input_ids = torch.cat(
                        [
                            chunk_input_ids,
                            torch.full(
                                (1, padding_length),
                                self.tokenizer.pad_token_id,
                                dtype=torch.long,
                            ),
                        ],
                        dim=1,
                    )
                    chunk_attention_mask = torch.cat(
                        [
                            chunk_attention_mask,
                            torch.zeros((1, padding_length), dtype=torch.long),
                        ],
                        dim=1,
                    )

                chunk_labels = chunk_input_ids.clone()
                chunk_labels[:, :-1] = chunk_input_ids[:, 1:]
                chunk_labels[:, -1] = self.tokenizer.pad_token_id

                all_input_ids.append(chunk_input_ids)
                all_attention_masks.append(chunk_attention_mask)
                all_labels.append(chunk_labels)

        # Concatenate chunks to create tensors
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        dataset_dict = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }

        self.dataset = Dataset.from_dict(dataset_dict)

    def preprocess(self):
        """Preprocesses the data and creates the Hugging Face Dataset."""
        self._process_data()
        return self.dataset
