import json
import os
import sys
import torch
from src.tasks.simulation_tasks import generate_stage_two_geometry

def main():
    print("Running generate_stage_two_geometry (local execution)...")
    
    try:
        results_geometry = generate_stage_two_geometry(n_points=100)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print("Geometry generation complete.")
    
    # Load existing results to preserve other stages
    output_file = 'results/theory_simulation.json'
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            full_results = json.load(f)
    else:
        full_results = {}
        
    full_results['stage_two_geometry'] = results_geometry
    
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=4)
        
    print(f"Updated {output_file} with new geometry results.")

if __name__ == "__main__":
    main()
