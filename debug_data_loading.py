
import sys
import traceback
import os
import torch

# Force CPU
torch.device = lambda x: torch.device('cpu') if isinstance(x, str) else x

try:
    # Add current dir to path
    sys.path.append(os.getcwd())
    from src.data.datasets import _load_medmnist_data
    print("Function imported.")
    loader, dataset = _load_medmnist_data('pathmnist', batch_size=32)
    print("Data loaded successfully.")
    print(f"Dataset length: {len(dataset)}")
    batch = next(iter(loader))
    print(f"Batch shape: {batch[0].shape}")
except Exception:
    traceback.print_exc()
