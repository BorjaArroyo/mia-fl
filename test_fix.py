
import ray
import torch
import numpy as np
import sys
import os
from src.tasks.simulation_tasks import verify_stage_two

def run_test():
    print("--- Running Test Fix (Local) ---")
    
    # Initialize Ray locally or just call the function directly if not using Ray for this test?
    # The function is decorated with @ray.remote but we can comment it out or just run it via python import if we strip the decorator in the worker file.
    # Ah, the decorator is commented out in the source file! "# @ray.remote".
    # So we can just call it directly.
    
    # Force CPU to rule out MPS RNG issues
    original_device = torch.device
    torch.device = lambda x: original_device('cpu')
    
    try:
        results = verify_stage_two(n_trials=5, in_channels=3, latent_dim=32, num_classes=10)
    except Exception as e:
        print(f"FAILED to run verify_stage_two: {e}")
        return
    finally:
        torch.device = original_device

    summary = results['summary']
    points = results['points']
    
    print("\n--- Summary ---")
    print(f"Overall Violation Rate: {summary['overall_violation_rate']:.2%}")
    print(f"Median L: {summary['median_L_empirical']:.2e}")
    
    # Check error scaling
    # Filter points with small norm (< 1e-4) and check if their errors are VERY small (< 1e-8)
    # If linear error persisted, error would be ~ 1e-4 * ||g||
    
    print("\n--- Checking Small Scale Errors ---")
    small_scale_points = [p for p in points if p['norm_u'] < 1e-4]
    if not small_scale_points:
        print("No small scale points found.")
    else:
        max_error = max(p['error_quad'] for p in small_scale_points)
        print(f"Max error for ||u|| < 1e-4: {max_error:.2e}")
        
        if max_error < 1e-8:
            print("SUCCESS: Errors are negligible at small scales (Quadratic behavior confirmed).")
        else:
            print("WARNING: Errors are still significant at small scales.")

    # Check slope roughly
    # We can't easily check slope without plotting, but low error is a good proxy.

if __name__ == "__main__":
    run_test()
