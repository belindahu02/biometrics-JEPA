from plot_results import *
import tensorflow as tf
import os

# Configure GPU memory to prevent OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Allow memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected, using CPU")

# single layer: ft=5
# 2 layer: ft=4
# 3 layer: ft=3
# 4 layer: ft=2
# all layer: ft=0

# Run experiments for different fine-tuning configurations
for layers in [0, 1, 2, 3, 4, 5]:
    print(f"\n{'='*60}")
    print(f"Running experiments for ft={layers}")
    print(f"{'='*60}\n")
    plotspu(layers)