import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Baseline (FP32)', 'Whole-model Quantization', 'Selective Quantization']
model_sizes = [474.7, 119.02, 231.70]
latencies = [11.74, 11.76, 11.59]
perplexities = [49.27, 57.51, 56.82]

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Model size plot
ax1.bar(models, model_sizes)
ax1.set_title('Model Size (MB)')
ax1.set_xlabel('Model')
ax1.set_ylabel('Size (MB)')
ax1.tick_params(axis='x', rotation=45)

# Latency and perplexity plot
x = np.arange(len(models))
width = 0.35
ax2.bar(x - width/2, latencies, width, label='Latency (ms)')
ax2.bar(x + width/2, perplexities, width, label='Perplexity')
ax2.set_title('Latency and Perplexity')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.tick_params(axis='x', rotation=45)
ax2.set_xlabel('Model')
ax2.set_ylabel('Value')
ax2.legend()

plt.tight_layout()
plt.savefig("quantize_gpt2.png")
plt.show()