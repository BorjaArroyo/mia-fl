"""Data partitioning utilities for Federated Learning simulations."""
import sys
import os
import torch
from .datasets import _load_medmnist_data

def get_fl_partitioned_data(dataset_name: str, n_clients: int, scenario: str, num_classes: int, limit: int = 2000) -> tuple[torch.Tensor, int, list]:
    """Load MedMNIST data and partition it for FL clients.
    
    Args:
        dataset_name: Name of dataset (e.g. 'pathmnist')
        n_clients: Number of clients
        scenario: 'iid', 'non-iid', or 'inverted'
        num_classes: Number of classes (for one-hot encoding if needed)
        limit: Max number of samples to load (default: 2000)
        
    Returns:
        X_4d: Full dataset tensor [N, C, H, W]
        num_classes_actual: Actual number of classes found
        client_data: List of (X_client, y_client) tuples
    """
    try:
        print(f"Loading {dataset_name} for scenario {scenario}...")
        loader, _ = _load_medmnist_data(dataset_name, batch_size=1000, shuffle=False, train=True, num_workers=0)
    except Exception as e:
        raise RuntimeError(f"MedMNIST data loading failed: {e}. Cannot proceed without real data.") from e
    
    all_X = []
    all_y = []
    for X, y in loader:
        all_X.append(X)
        all_y.append(y)
        
    X_full = torch.cat(all_X)
    y_full = torch.cat(all_y).flatten().long()
    
    # Limit dataset size for simulation speed
    X_full = X_full[:limit]
    y_full = y_full[:limit]
    
    X_4d = X_full.float() / 255.0
    if X_4d.dim() == 3: 
        X_4d = X_4d.unsqueeze(1)
    
    client_data = []
    perm = torch.randperm(len(X_4d))
    
    if scenario == 'iid':
        chunk_size = len(X_4d) // n_clients
        for i in range(n_clients):
            idx = perm[i*chunk_size : (i+1)*chunk_size]
            client_data.append((X_4d[idx], y_full[idx]))
            
    elif scenario == 'non-iid':
        sorted_idx = torch.argsort(y_full)
        X_sorted = X_4d[sorted_idx]
        y_sorted = y_full[sorted_idx]
        chunk_size = len(X_4d) // n_clients
        for i in range(n_clients):
            idx = slice(i*chunk_size, (i+1)*chunk_size)
            client_data.append((X_sorted[idx], y_sorted[idx]))
            
    elif scenario == 'inverted':
        chunk_size = len(X_4d) // n_clients
        for i in range(n_clients):
            idx = perm[i*chunk_size : (i+1)*chunk_size]
            X_c = X_4d[idx]
            y_c = y_full[idx]
            # Invert images for all clients except the target (Client 0)
            if i != 0:
                X_c = 1.0 - X_c 
            client_data.append((X_c, y_c))
            
    return X_4d, len(torch.unique(y_full)), client_data
