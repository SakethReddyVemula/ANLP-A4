import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import time
from typing import Dict, List, Optional, Union, Tuple
import os

class Int8Tensor:
    def __init__(self, tensor: torch.Tensor, scale: float):
        self.scale = scale
        self.quantized_data = torch.round(tensor * scale).to(torch.int8)
        
    def dequantize(self) -> torch.Tensor:
        return self.quantized_data.float() / self.scale
    
    def element_size(self) -> int:
        return 1  # int8 = 1 byte
    
    def nelement(self) -> int:
        return self.quantized_data.nelement()

class ModelQuantizer:
    def __init__(self, model_name: str = 'gpt2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.quantized_params = {}
        
    def get_model_size_mb(self) -> Tuple[float, str]:
        total_bytes = 0
        total_params = 0
        quantized_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += 1
            if name in self.quantized_params:
                quantized_params += 1
                param_bytes = (self.quantized_params[name].nelement() * 
                             self.quantized_params[name].element_size())
            else:
                param_bytes = param.nelement() * param.element_size()
            total_bytes += param_bytes
        
        size_mb = total_bytes / (1024 * 1024)
        details = f"({quantized_params}/{total_params} params quantized)"
        return size_mb, details

    def quantize_tensor(self, tensor: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, Int8Tensor]:
        max_val = torch.max(torch.abs(tensor))
        scale = (2 ** (num_bits - 1) - 1) / max_val
        
        quantized_storage = Int8Tensor(tensor, scale)
        return quantized_storage.dequantize(), quantized_storage

    def quantize_whole_model(self, num_bits: int = 8):
        self.quantized_params = {}
        
        # Quantize transformer weights
        for name, param in self.model.transformer.named_parameters():
            if param.dim() > 1:  # Only quantize weight matrices, not bias vectors
                with torch.no_grad():
                    dequantized, quantized = self.quantize_tensor(param.data, num_bits)
                    param.data = dequantized
                    self.quantized_params[f"transformer.{name}"] = quantized
        
        # Quantize embedding weights
        emb_name = "transformer.wte.weight"
        with torch.no_grad():
            dequantized, quantized = self.quantize_tensor(
                self.model.transformer.wte.weight.data, num_bits)
            self.model.transformer.wte.weight.data = dequantized
            self.quantized_params[emb_name] = quantized
        
        # Quantize LM head weights
        lm_head_name = "lm_head.weight"
        if hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'weight'):
            with torch.no_grad():
                dequantized, quantized = self.quantize_tensor(
                    self.model.lm_head.weight.data, num_bits)
                self.model.lm_head.weight.data = dequantized
                self.quantized_params[lm_head_name] = quantized

    def quantize_selective(self, components: List[str], num_bits: int = 8):
        self.quantized_params = {}
        
        component_mapping = {
            'ffn': ['mlp.c_fc', 'mlp.c_proj'],
            'attention': ['attn.c_attn', 'attn.c_proj'],
            'query': ['attn.c_attn'],  # Note: In GPT-2, Q,K,V are combined in c_attn
            'key': ['attn.c_attn'],
            'value': ['attn.c_attn']
        }
        
        # Expand components to their corresponding layer patterns
        patterns = []
        for component in components:
            if component in component_mapping:
                patterns.extend(component_mapping[component])
            else:
                patterns.append(component)
        
        # Quantize matching parameters
        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # Only quantize weight matrices
                if any(pattern in name for pattern in patterns):
                    with torch.no_grad():
                        dequantized, quantized = self.quantize_tensor(param.data, num_bits)
                        param.data = dequantized
                        self.quantized_params[name] = quantized

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

def evaluate_model(quantizer: ModelQuantizer):
    print("\nEvaluation Metrics:")
    print("-" * 50)
    
    model_size, size_details = quantizer.get_model_size_mb()
    print(f"Model Size: {model_size:.2f} MB {size_details}")
    
    mean_latency, std_latency = quantizer.measure_inference_latency()
    print(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")    
    try:
        perplexity = quantizer.compute_perplexity()
        print(f"Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error computing perplexity: {str(e)}")

def save_model(quantizer: ModelQuantizer, save_path: str, save_name: str):
    torch.save(quantizer.quantized_params, os.path.join(save_path, f"{save_name}.pth"))
    torch.save(quantizer.model.state_dict(), os.path.join(save_path, "model.pth"))
    quantizer.tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")

def load_model(quantizer: ModelQuantizer, load_path: str, save_name: str):
    quantizer.quantized_params = torch.load(os.path.join(load_path, f"{save_name}.pth"))
    quantizer.model.load_state_dict(torch.load(os.path.join(load_path, "model.pth")))
    quantizer.model.to(quantizer.device)
    quantizer.tokenizer = GPT2Tokenizer.from_pretrained(load_path)
    print(f"Model loaded from: {load_path}")

def main():
    save_path = "./saved_models/"
    load_path = "./saved_models/"

    # Initialize quantizer
    print("Loading baseline model...")
    quantizer = ModelQuantizer('gpt2')
    
    # Baseline metrics
    print("Computing baseline metrics...")
    evaluate_model(quantizer)
    
    # Whole model quantization
    print("\nPerforming whole-model quantization...")
    quantizer.quantize_whole_model(num_bits=8)
    print("\nWhole-Model Quantization Metrics:")
    evaluate_model(quantizer)

    save_model(quantizer, save_path, "whole_model_quantized")

    # Loading and testing the quantized model
    quantizer = ModelQuantizer('gpt2')
    load_model(quantizer, load_path, "whole_model_quantization")
    evaluate_model(quantizer)
    
    # Reset model for selective quantization
    print("\nResetting model for selective quantization...")
    quantizer = ModelQuantizer('gpt2')
    
    # Selective quantization of FFN and attention components
    print("\nPerforming selective quantization...")
    components_to_quantize = ['ffn', 'attention']  # Quantize FFN and attention components
    quantizer.quantize_selective(components_to_quantize, num_bits=8)
    print("\nSelective Quantization Metrics:")
    evaluate_model(quantizer)

    save_model(quantizer, save_path, "selective_quantization")
    
    # Load and test model 
    quantizer = ModelQuantizer('gpt2')
    load_model(quantizer, load_path, "selective_quantization")
    evaluate_model(quantizer)
    
if __name__ == "__main__":
    main()