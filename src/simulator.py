import json
import os
import ray
from src.tasks.simulation_tasks import (
    verify_stage_one,
    verify_stage_two,
    generate_stage_two_geometry,
    simulate_trajectory,
)

# Wrap tasks with ray.remote and ensure they run with 'uv run'
# This ensures dependencies are installed/used on the worker
UV_RUNTIME_ENV = {"py_executable": "uv run"}

verify_stage_one_remote = ray.remote(runtime_env=UV_RUNTIME_ENV)(verify_stage_one)
# VAE-heavy tasks should request GPU resources
verify_stage_two_remote = ray.remote(num_gpus=1, runtime_env=UV_RUNTIME_ENV)(verify_stage_two)
generate_stage_two_geometry_remote = ray.remote(num_gpus=1, runtime_env=UV_RUNTIME_ENV)(generate_stage_two_geometry)
simulate_trajectory_remote = ray.remote(num_gpus=1, runtime_env=UV_RUNTIME_ENV)(simulate_trajectory)

class TheorySimulator:
    def __init__(self, output_file='results/theory_simulation.json'):
        self.output_file = output_file
        self.results = {}
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        
    def run(self, address=None, **kwargs):
        """Run the simulation.
        
        Args:
            address: Ray cluster address (e.g. "ray://192.168.0.158:10001"). 
                     If None, connects to local Ray instance.
            **kwargs: Additional args passed to tasks (e.g. limit)
        """
        if address:
            print(f"Connecting to Ray cluster at {address}...")
            # Sync working directory to cluster
            # Exclude heavy/generated files to speed up sync
            # Use leading slash to anchor to root and avoid excluding src/data
            ray.init(address=address, runtime_env={
                "working_dir": ".",
                "excludes": [".venv", "__pycache__", "results", "/data", "figs", ".git", "temp_log.txt"]
            })
        else:
            print("Connecting to local Ray instance...")
            ray.init()
            
        print("Connected to Ray.")

        TIMEOUT_SECONDS = 3600  # 1 hour per task
        
        task_order = [
            ('stage_one', verify_stage_one_remote),
            ('stage_two', verify_stage_two_remote),
            ('stage_two_geometry', generate_stage_two_geometry_remote),
            ('trajectory', simulate_trajectory_remote),
        ]
        
        
        print(f"Launching {len(task_order)} tasks sequentially...")
        for name, task_fn in task_order:
            print(f"  Launching '{name}'...")
            if name == 'trajectory':
                 future = task_fn.remote(limit=kwargs.get('limit', 2000))
            else:
                 future = task_fn.remote()
            try:
                # Synchronous wait for results
                # We use a timeout to prevent hanging indefinitely
                self.results[name] = ray.get(future, timeout=TIMEOUT_SECONDS)
                print(f"  Task '{name}' completed successfully.")
            except TimeoutError:
                print(f"  Task '{name}' TIMED OUT after {TIMEOUT_SECONDS}s.")
                self.results[name] = {'error': f'Timed out after {TIMEOUT_SECONDS}s'}
                ray.cancel(future, force=True)
            except Exception as e:
                print(f"  Task '{name}' FAILED: {e}")
                self.results[name] = {'error': str(e)}
        
        n_ok = sum(1 for v in self.results.values() if not isinstance(v, dict) or 'error' not in v)
        print(f"\n{n_ok}/{len(task_order)} tasks completed successfully.")
        
        # We don't need to print the whole JSON to stdout if it's large, 
        # but let's keep the marker for the user's scripts if they rely on it.
        print("START_RESULTS_JSON")
        # For very large results, we might only print a summary
        print(json.dumps({k: "..." if isinstance(v, dict) and len(str(v)) > 1000 else v for k, v in self.results.items()}))
        print("END_RESULTS_JSON")

        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {self.output_file}")
        
        ray.shutdown()
