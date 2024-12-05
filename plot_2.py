import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Baseline (FP32)', '8-bit Quantization', '4-bit FP4 Quantization', '4-bit NF4 Quantization']
model_sizes = [474.7, 118.68, 39.09, 39.09]
latencies = [11.7, 55.63, 24.22, 24.05]
perplexities = [49.27, 49.5, 57.72, 52.91]

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
plt.savefig("bitsandbytes.png")
plt.show()