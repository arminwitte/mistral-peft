import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import PyPDF2
import spacy
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from markitdown import MarkItDown
import nltk

# try:
#     nltk.data.find('tokenizers/punkt_tab')  # Check if Punkt tokenizer is downloaded
# except LookupError:
#     print("Downloading Punkt tokenizer. This might take a moment...")
nltk.download('punkt_tab')


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


################### SENTENCE EXTRACTOR ###################






class TextExtractor:
    def __init__(self, language: str = "en"):
        """Initialize spaCy for sentence extraction.
        
        Args:
            language: 'en' for English or 'de' for German
        """
        model_name = "en_core_web_sm" if language == "en" else "de_core_news_sm"
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
            
        # Configure pipeline for speed
        self.nlp.select_pipes(enable=["senter", "parser"])

    def markitdown(self, file_path: Union[str, Path]) -> str:
        md = MarkItDown()
        result = md.convert(file_path)
        return result.text_content
            
    def read_pdf(self, file_path: Union[str, Path]) -> str:
        """Read text from PDF file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
            
        text = []
        with open(path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            for page in tqdm(pdf.pages, desc="Reading PDF"):
                text.append(page.extract_text())
                
        return "\n".join(text)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove page numbers and headers
        text = re.sub(r'\f', '\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'(?i)(page|seite)\s*\d+\s*(?:of|von)\s*\d+', '', text)
        
        # Remove references
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract valid sentences from text."""
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            sentence = sent.text.strip()
            # Basic validation
            if len(sentence.split()) >= 3 and len(sentence) >= 10:
                sentences.append(sentence)
                
        return sentences
    
    def split_into_sentences(self, text, markdown_aware=True):
        """Splits a text (optionally Markdown) into sentences.

        Args:
            text: The input text string.
            markdown_aware: If True, attempts to handle some basic Markdown
                        syntax to avoid splitting within links or emphasis.

        Returns:
            A list of strings, where each string is a sentence.
            Returns an empty list if the input text is None or empty.
        """

        if not text:  # Handle None or empty input
            return []

        if markdown_aware:
            # Protect Markdown links and emphasis from splitting
            text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"LINK_\1_TO_\2", text)  # Links
            text = re.sub(r"\*\*([^*]+)\*\*", r"BOLD_\1", text)  # Bold text
            text = re.sub(r"\*([^*]+)\*", r"ITALIC_\1", text)  # Italic text
            text = re.sub(r"`([^`]+)`", r"CODE_\1", text) # Inline code

        sentences = nltk.sent_tokenize(text)

        if markdown_aware:
            # Restore Markdown syntax (this is a basic approach and might need refinement)
            sentences = [s.replace("LINK_", "[").replace("_TO_", "](").replace("BOLD_", "**").replace("ITALIC_", "*").replace("CODE_", "`") for s in sentences]

        return sentences

    def split_chunks(self, text, chunk_size=4096):
        n_chunks = len(text) // chunk_size
        chunks = []
        for i in range(n_chunks):
            chunks.append(text[i*chunk_size:(i+1)*chunk_size])

        return chunks
       
    
    def process_document(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """Process PDF document and extract sentences.
        
        Args:
            file_path: Path to PDF file
            output_path: Optional path to save JSON output
            
        Returns:
            Dict with metadata and extracted sentences
        """
        # Extract and process text
        raw_text = self.read_pdf(file_path)
        # raw_text = self.markitdown(file_path)
        print(f"Raw text length: {len(raw_text)}")
        clean_text = self.clean_text(raw_text)
        print(f"Cleaned text length: {len(raw_text)}")
        # sentences = self.extract_sentences(clean_text)
        sentences = self.split_chunks(clean_text)
        print(f"Sentences: {len(sentences)}")
        
        # Prepare output
        output = {
            "metadata": {
                "source": str(file_path),
                "language": self.nlp.lang,
                "num_sentences": len(sentences)
            },
            "sentences": sentences
        }
        
        # Save if requested
        if output_path:
            path = Path(output_path)
            with path.open('w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
                
        return output

    # Example usage
    # extractor = SentenceExtractor()
    # try:
    #     result = extractor.process_document("example.pdf", "sentences.json")
    #     print(f"Extracted {result['metadata']['num_sentences']} sentences")
    # except FileNotFoundError as e:
    #     print(f"Error: {e}")


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
