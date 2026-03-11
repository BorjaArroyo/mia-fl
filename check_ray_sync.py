import argparse
import ray

# Use uv run to ensure dependencies are installed and used
@ray.remote(runtime_env={"py_executable": "uv run"})
def get_env_info():
    import os
    import sys
    import subprocess
    info = {
        "node": os.uname().nodename,
        "pid": os.getpid(),
        "python": sys.version,
        "working_dir": os.getcwd(),
        "files_in_cwd": os.listdir('.'),
        "env_vars": {k: v for k, v in os.environ.items() if "RAY" in k or "UV" in k}
    }
    
    try:
        info["uv_version"] = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()
    except Exception:
        info["uv_version"] = "not found"
        
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"
        
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Ray + uv Connectivity")
    parser.add_argument("--address", type=str, default=None, 
                        help="Ray cluster address (e.g. 'ray://192.168.0.158:10001'). Default: None (local)")
    args = parser.parse_args()

    if args.address:
        print(f"Connecting to Ray cluster at {args.address}...")
        ray.init(address=args.address, runtime_env={
            "working_dir": ".",
            "excludes": [".venv", "__pycache__", "results", "/data", "figs", ".git", "temp_log.txt"]
        })
    else:
        print("Connecting to local Ray instance...")
        ray.init()
        
    print("Connected to Ray. Fetching worker info...")
    
    try:
        info = ray.get(get_env_info.remote())
        print("\nRemote Worker Environment:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        
        if info.get("torch_version") != "NOT INSTALLED":
            print("\n✓ torch is available — uv dependency resolution works!")
        else:
            print("\n✗ torch NOT installed — uv may not be resolving dependencies")
            
    except Exception as e:
        print(f"\n✗ Error executing task on cluster: {e}")
    
    ray.shutdown()
