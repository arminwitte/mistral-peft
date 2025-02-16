import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



def load_base_model(model_name: str = "mistralai/Mistral-7B-v0.3"):
    """Load the base model and tokenizer with 4-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Use load_in_8bit=True for 8-bit quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    quantization_config=quantization_config,
    )
    return model, tokenizer


def prepare_for_training(
    model, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05
):
    """Prepare model for LoRA training."""
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model


def prepare_text_chunks(text: str, max_length: int = 2000) -> List[str]:
    """Split text into chunks for training."""
    chunks = []
    current_chunk = ""

    for paragraph in text.split("\n\n"):
        if len(current_chunk) + len(paragraph) < max_length:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs,
) -> str:
    """Generate response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[
        len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :
    ]

    return response.strip()


################### TEXT EXTRACTOR ###################


import re
import json
import httpx
import tempfile
import os
from pathlib import Path
from typing import Union, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pypdf import PdfReader

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
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp.write(response.content)
        tmp.close()
        return Path(tmp.name)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text for LLM finetuning."""
        # Add this before other cleaning steps
        # Look for common patterns of glued words like capitalLetters
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Split "wordWord" into "word Word"
        text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)  # Split "word123" into "word 123"
        text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)  # Split "123word" into "123 word"

        # Remove headers and footers
        text = re.sub(r'\f', '\n', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
        text = re.sub(r'(?i)(page|seite)\s*\d+\s*(?:of|von)\s*\d+', '', text)
        
        # Clean scientific formatting
        text = re.sub(r'(?<=\d)\.(?=\d)', '.0', text)  # Add leading 0 after decimal
        text = re.sub(r'×10\^(-?\d+)', lambda m: f'e{m.group(1)}', text)  # Fix scientific notation
        
        # Clean references and citations
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)  # Remove citation brackets
        text = re.sub(r'\(\d{4}\)', '', text)  # Remove year citations
        text = re.sub(r'et al\.', 'et al', text)  # Standardize et al
        
        # Fix paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Standardize paragraph breaks
        text = re.sub(r'([.!?])\n(?=[A-Z])', r'\1\n\n', text)  # Add breaks after sentences
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        # Remove redundant whitespace around punctuation
        text = re.sub(r'\s+([.,;?!])', r'\1', text)
        text = re.sub(r'"\s+([^"]+)\s+"', r'"\1"', text)
        
        return text.strip()

    def process_single_document(
        self,
        source: Union[str, Path],
        is_url: bool = False
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
                    "is_url": is_url
                },
                "text": clean_text
            }
            
        except Exception as e:
            return {
                "metadata": {
                    "source": str(source),
                    "error": str(e),
                    "is_url": is_url
                },
                "text": ""
            }

    def process_documents(
        self,
        sources: List[Union[str, Path]],
        url_list: bool = False
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
            
            for future in tqdm(futures, total=len(sources), desc="Processing documents"):
                self.results.append(future.result())

    def __enter__(self):
        """Context manager entry."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit. Saves results and cleans up."""
        try:
            if self.results:
                with self.output_file.open('w', encoding='utf-8') as f:
                    json.dump({
                        "documents": self.results,
                        "total_documents": len(self.results),
                        "successful_documents": sum(1 for doc in self.results if not doc["metadata"].get("error"))
                    }, f, ensure_ascii=False, indent=2)
        finally:
            self.http_client.close()




import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

class CLAPreprocessor:
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
            with open(file_path, 'r', encoding='utf-8') as f:
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
            tokenized_sentence = self.tokenizer(sentence, truncation=False, padding=False, return_tensors="pt") # No padding/truncation here
            input_ids = tokenized_sentence["input_ids"]
            attention_mask = tokenized_sentence["attention_mask"]

            # Create chunks
            for i in range(0, input_ids.size(1), chunk_size -1): # Changed range to chunk_size-1 as we want to shift by one
                chunk_input_ids = input_ids[:, i:i + chunk_size]
                chunk_attention_mask = attention_mask[:, i:i + chunk_size]

                # Ensure chunk is of correct length by padding if necessary
                padding_length = chunk_size - chunk_input_ids.size(1)
                if padding_length > 0:
                  chunk_input_ids = torch.cat([chunk_input_ids, torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1)
                  chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)

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
            "labels": all_labels
        }

        self.dataset = Dataset.from_dict(dataset_dict)


    def preprocess(self):
        """Preprocesses the data and creates the Hugging Face Dataset."""
        self._process_data()
        return self.dataset












################### MLM PREPROCESSOR ###################


class MLMPreprocessor:
    def __init__(
        self,
        tokenizer_name: str = "mistralai/Mistral-7B-v0.3",
        mask_probability: float = 0.15,
        max_length: int = 512,
        seed: Optional[int] = None,
    ):
        """Initialize MLM preprocessor.

        Args:
            tokenizer_name: HuggingFace tokenizer name/path
            mask_probability: Probability of masking each token
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.mask_probability = mask_probability
        self.max_length = max_length

        if seed is not None:
            random.seed(seed)

    def load_sentences(self, file_path: Union[str, Path]) -> List[str]:
        """Load sentences from JSON file.

        Args:
            file_path: Path to JSON file with extracted sentences

        Returns:
            List of sentences
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data["sentences"]

    def create_mlm_example(self, sentence: str) -> Dict:
        """Create a single MLM training example.

        Args:
            sentence: Input sentence

        Returns:
            Dict with input_ids, attention_mask, and labels
        """
        # Tokenize
        tokens = self.tokenizer(
            sentence, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        # Create masks
        input_ids = tokens.input_ids[0].tolist()
        masked_input_ids = input_ids[:]
        labels = [-100] * len(masked_input_ids)  # -100 is the ignore index

        # # Randomly mask tokens
        # for idx in range(1, len(input_ids) - 1):  # Skip special tokens
        #     if random.random() < self.mask_probability:
        #         labels[idx] = input_ids[idx]  # Save original token
        #         input_ids[idx] = self.tokenizer.mask_token_id




        # 1. Random Masking (as before)
        for i in range(len(masked_input_ids)):
            if random.random() < self.mask_probability:
                # print(f"{i:8d}: {masked_input_ids[i]} -> {self.tokenizer.unk_token_id}")
                labels[i] = masked_input_ids[i]
                masked_input_ids[i] = self.tokenizer.unk_token_id

        # # 2. Enforce at least one mask (NEW)
        # masked_indices = [i for i, token_id in enumerate(masked_input_ids) if token_id == self.tokenizer.mask_token_id]

        # if not masked_indices: # If NO tokens were masked
        #     random_index = random.randint(0, len(input_ids) - 1) # Choose a random index
        #     labels[random_index] = input_ids[random_index] # Set the label
        #     masked_input_ids[random_index] = self.tokenizer.mask_token_id # Mask it

        return {
            "input_ids": masked_input_ids,
            "attention_mask": tokens.attention_mask[0].tolist(),
            "labels": labels,
        }


    def prepare_dataset(
        self, sentences: List[str], train_split: float = 0.9
    ) -> Dict[str, Dataset]:
        """Create MLM datasets from sentences.

        Args:
            sentences: List of input sentences
            train_split: Fraction of data to use for training

        Returns:
            Dict with 'train' and 'val' datasets
        """
        examples = []
        for sentence in tqdm(sentences, desc="Creating MLM examples"):
            examples.append(self.create_mlm_example(sentence))

        # Split into train/val
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        return {
            "train": Dataset.from_list(train_examples),
            "val": Dataset.from_list(val_examples),
        }

    def process_files(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        train_split: float = 0.9,
    ) -> Dict[str, Dataset]:
        """Process multiple input files and create datasets.

        Args:
            input_files: List of paths to JSON files with sentences
            output_dir: Optional directory to save processed datasets
            train_split: Fraction of data to use for training

        Returns:
            Dict with 'train' and 'val' datasets
        """
        # Collect all sentences
        all_sentences = []
        for file_path in input_files:
            sentences = self.load_sentences(file_path)
            all_sentences.extend(sentences)

        # Create datasets
        datasets = self.prepare_dataset(all_sentences, train_split)

        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for split, dataset in datasets.items():
                dataset.save_to_disk(output_dir / split)

        return datasets

    # Example usage
    # preprocessor = MLMPreprocessor(
    #     mask_probability=0.15,
    #     max_length=512,
    #     seed=42
    # )

    # try:
    #     # Process single file
    #     sentences = preprocessor.load_sentences("sentences.json")
    #     datasets = preprocessor.prepare_dataset(sentences)
    #     print(f"Created datasets with {len(datasets['train'])} training and "
    #           f"{len(datasets['val'])} validation examples")

    #     # Or process multiple files
    #     datasets = preprocessor.process_files(
    #         ["file1.json", "file2.json"],
    #         output_dir="processed_data"
    #     )

    # except FileNotFoundError as e:
    #     print(f"Error: {e}")


################### EXPORT TO OLLAMA ###################


def export_to_ollama(
    model_path: Union[str, Path],
    model_name: str,
    description: str = "",
    license: str = "Apache 2.0",
    tags: Optional[list] = None,
    system_prompt: str = "",
) -> None:
    """Export a fine-tuned model to Ollama format.

    Args:
        model_path: Path to the fine-tuned model directory
        model_name: Name for the Ollama model
        description: Model description
        license: Model license
        tags: List of tags for the model
        system_prompt: System prompt to use with the model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Create temporary directory for conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # First convert to GGUF format using llama.cpp
        print("Converting to GGUF format...")
        gguf_path = temp_path / "model.gguf"
        convert_to_gguf(model_path, gguf_path)

        # Create Modelfile
        modelfile_content = create_modelfile(
            model_name=model_name,
            description=description,
            license=license,
            tags=tags or [],
            system_prompt=system_prompt,
        )

        modelfile_path = temp_path / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        # Create Ollama model
        print(f"Creating Ollama model: {model_name}")
        create_ollama_model(modelfile_path, gguf_path, model_name)

        print(f"Model '{model_name}' has been created in Ollama")
        print(f"You can now use it with: ollama run {model_name}")


def convert_to_gguf(input_path: Path, output_path: Path, num_threads: int = 4) -> None:
    """Convert model to GGUF format using llama.cpp."""
    try:
        # Check if llama-cpp-python is installed
        subprocess.run(
            ["python", "-c", "import llama_cpp"], check=True, capture_output=True
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "llama-cpp-python not found. Install with: pip install llama-cpp-python"
        )

    cmd = [
        "python",
        "-m",
        "llama_cpp.convert",
        "--outfile",
        str(output_path),
        "--outtype",
        "q4_k_m",  # 4-bit quantization
        "--threads",
        str(num_threads),
        str(input_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")


def create_modelfile(
    model_name: str,
    description: str = "",
    license: str = "Apache 2.0",
    tags: list = None,
    system_prompt: str = "",
) -> str:
    """Create Ollama Modelfile content."""
    tags = tags or []

    content = [
        f"FROM {model_name}.gguf",
        f'DESCRIPTION "{description}"',
        f'LICENSE "{license}"',
        *[f'TAG "{tag}"' for tag in tags],
    ]

    if system_prompt:
        content.append(f'SYSTEM """{system_prompt}"""')

    # Add some parameter configurations
    content.extend(
        [
            "PARAMETER stop ",
            "PARAMETER temperature 0.7",
            "PARAMETER top_k 40",
            "PARAMETER top_p 0.9",
            "PARAMETER repeat_penalty 1.1",
        ]
    )

    return "\n".join(content)


def create_ollama_model(modelfile_path: Path, gguf_path: Path, model_name: str) -> None:
    """Create Ollama model from Modelfile and GGUF model."""
    # Create model directory in Ollama's model path
    ollama_path = Path.home() / ".ollama" / "models" / model_name
    ollama_path.mkdir(parents=True, exist_ok=True)

    # Copy GGUF model
    shutil.copy2(gguf_path, ollama_path / f"{model_name}.gguf")

    # Copy Modelfile
    shutil.copy2(modelfile_path, ollama_path / "Modelfile")

    # Create Ollama model
    cmd = ["ollama", "create", model_name]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create Ollama model:\n{result.stderr}")

    # export_to_ollama(
    #     model_path="./final_model",
    #     model_name="dissertation-assistant",
    #     description="Fine-tuned model for technical engineering content",
    #     tags=["engineering", "technical", "heat-transfer"],
    #     system_prompt="""You are a technical assistant specialized in heat transfer
    #     and fluid dynamics. You provide accurate, technical responses based on
    #     engineering principles and research data."""
    # )
