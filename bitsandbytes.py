import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import bitsandbytes as bnb
from typing import Optional, Tuple, List
import numpy as np
import os

class BitsAndBytesQuantizer:
    def __init__(self, model_name: str = 'gpt2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.load_baseline_model()
    
    def load_baseline_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float32
        )
        
    def load_8bit_model(self):
        self.model = None
        torch.cuda.empty_cache()
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # print(self.model.get_memory_footprint())
        
    def load_4bit_model(self, use_nf4: bool = False):
        self.model = None
        torch.cuda.empty_cache()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4" if use_nf4 else "fp4"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
        )

        # print(self.model.get_memory_footprint())

    def get_model_size_mb(self) -> Tuple[float, str]:
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'quantization_config'):
            quant_config = self.model.config.quantization_config
            if isinstance(quant_config, BitsAndBytesConfig):
                if quant_config.load_in_4bit:
                    param_size = 0.5  # 4 bits = 0.5 bytes
                    quant_type = "4-bit"
                elif quant_config.load_in_8bit:
                    param_size = 1  # 8 bits = 1 byte
                    quant_type = "8-bit"
                else:
                    param_size = 4  # assuming fp32
                    quant_type = "fp32"
            else:
                param_size = 4
                quant_type = "fp32"
        else:
            param_size = 4
            quant_type = "fp32"
            
        size_mb = (total_params * param_size) / (1024 * 1024)
        return size_mb, f"({quant_type} quantization)"

    def measure_inference_latency(self, 
                            dataset_name: str = 'wikitext',
                            dataset_config: str = 'wikitext-2-raw-v1',
                            split: str = 'test',
                            max_samples: int = 100,
                            warmup_runs: int = 10) -> Tuple[float, float]:

        dataset = load_dataset(dataset_name, dataset_config, split=split)
        dataset = dataset.select(range(min(len(dataset), max_samples)))
        processed_inputs = []
        for item in dataset:
            if not item['text'].strip():
                continue
                
            inputs = self.tokenizer(item['text'],
                                return_tensors='pt',
                                truncation=True,
                                max_length=512)
            
            if inputs['input_ids'].size(1) == 0:
                continue
                
            processed_inputs.append(inputs.to(self.device))
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for inputs in processed_inputs[:5]:  # Use subset for warmup
                    self.model(**inputs)
        
        torch.cuda.synchronize()  # Ensure all operations are complete
        
        # Measure latencies
        latencies = []
        for inputs in processed_inputs:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                self.model(**inputs)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        return mean_latency, std_latency

    def compute_perplexity(self, dataset_name: str = 'wikitext',
                          dataset_config: str = 'wikitext-2-raw-v1',
                          split: str = 'test',
                          max_samples: int = 3000) -> float:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        dataset = dataset.select(range(min(len(dataset), max_samples)))
        
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for item in dataset:
                if not item['text'].strip():
                    continue
                    
                inputs = self.tokenizer(item['text'],
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=512).to(self.device)
                
                if inputs['input_ids'].size(1) == 0:
                    continue
                    
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        return torch.exp(torch.tensor(total_loss / total_tokens))
    
def save_model(quantizer: BitsAndBytesQuantizer, save_path: str, quantization_type: str):
    torch.save(quantizer.model.state_dict(), os.path.join(save_path, f"{quantization_type}_model.pth"))
    quantizer.tokenizer.save_pretrained(save_path)
    print(f"{quantization_type} model saved to: {save_path}")

def load_model(quantizer: BitsAndBytesQuantizer, load_path: str, quantization_type: str):
    if quantization_type == "fp32":
        quantizer.load_baseline_model()
    elif quantization_type == "8bit":
        quantizer.load_8bit_model()
    elif quantization_type == "4bit_fp4":
        quantizer.load_4bit_model(use_nf4=False)
    elif quantization_type == "4bit_nf4":
        quantizer.load_4bit_model(use_nf4=True)
    else:
        raise ValueError(f"Invalid quantization type: {quantization_type}")

    quantizer.model.load_state_dict(torch.load(os.path.join(load_path, f"{quantization_type}_model.pth")))
    quantizer.model.to(quantizer.device)
    quantizer.tokenizer = AutoTokenizer.from_pretrained(load_path)
    print(f"{quantization_type} model loaded from: {load_path}")

def evaluate_quantization(model_name: str = 'gpt2', 
                         input_text: str = "The quick brown fox jumps over the lazy dog"):
    print("\nQuantization Comparison")
    print("-" * 50)
    
    # Test baseline model
    print("\nBaseline Model (FP32):")
    quantizer = BitsAndBytesQuantizer(model_name)
    size, details = quantizer.get_model_size_mb()
    print(f"Model Size: {size:.2f} MB {details}")
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")
    
    # Test 8-bit quantization
    print("\n8-bit Quantization:")
    quantizer.load_8bit_model()
    size, details = quantizer.get_model_size_mb()
    print(f"Model Size: {size:.2f} MB {details}")
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")
    
    # Save the 8-bit quantized model
    save_path = "./saved_models/8bit_model"
    save_model(quantizer, save_path, "8bit")
    
    # Load the 8-bit quantized model
    quantizer = BitsAndBytesQuantizer(model_name)
    load_model(quantizer, save_path, "8bit")
    print("\nLoaded 8-bit Quantized Model:")
    size, details = quantizer.get_model_size_mb()
    print(f"Model Size: {size:.2f} MB {details}")
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")

    # Test 4-bit FP4 quantization
    print("\n4-bit FP4 Quantization:")
    quantizer.load_4bit_model(use_nf4=False)
    size, details = quantizer.get_model_size_mb()
    print(f"Model Size: {size:.2f} MB {details}")
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")
    
    # Save the 4-bit FP4 quantized model
    save_path = "./saved_models/4bit_fp4_model"
    save_model(quantizer, save_path, "4bit_fp4")
    
    # Load the 4-bit FP4 quantized model
    quantizer = BitsAndBytesQuantizer(model_name)
    load_model(quantizer, save_path, "4bit_fp4")
    print("\nLoaded 4-bit FP4 Quantized Model:")
    size, details = quantizer.get_model_size_mb()
    print(f"Model Size: {size:.2f} MB {details}")
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")

    # Test 4-bit NF4 quantization
    print("\n4-bit NF4 Quantization:")
    quantizer.load_4bit_model(use_nf4=True)
    size, details = quantizer.get_model_size_mb()
    print(f"Model Size: {size:.2f} MB {details}")
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")
    
    # Save the 4-bit NF4 quantized model
    save_path = "./saved_models/4bit_nf4_model"
    save_model(quantizer, save_path, "4bit_nf4")
    
    # Load the 4-bit NF4 quantized model
    quantizer = BitsAndBytesQuantizer(model_name)
    load_model(quantizer, save_path, "4bit_nf4")
    print("\nLoaded 4-bit NF4 Quantized Model:")
    size, details = quantizer.get_model_size_mb()
    print(f"Model Size: {size:.2f} MB {details}")
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")

if __name__ == "__main__":
    evaluate_quantization()