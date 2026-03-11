import os
import sys
import time
from ray.job_submission import JobSubmissionClient, JobStatus

DEFAULT_DASHBOARD_URL = "http://192.168.0.158:8265"

def get_client(address: str = DEFAULT_DASHBOARD_URL) -> JobSubmissionClient:
    """Get a Ray Job Submission Client."""
    return JobSubmissionClient(address)

def submit_and_wait(
    entrypoint: str = "uv run simulate_results.py",
    address: str = DEFAULT_DASHBOARD_URL,
) -> str:
    """Submit a job to the Ray cluster using uv + Ray integration and wait for completion.
    
    Uses py_executable='uv run' so all workers run under uv, which installs
    dependencies from pyproject.toml automatically. The working_dir is synced
    to the cluster so workers have access to pyproject.toml and uv.lock.
    
    This function streams logs in real-time and blocks until completion.
    
    Args:
        entrypoint: Command to run on the cluster (default: 'uv run simulate_results.py').
        address: Ray Dashboard URL.
    
    Returns:
        The job ID.
    """
    client = get_client(address)
    
    runtime_env = {
        "working_dir": os.getcwd(),
        "excludes": [".venv", ".git", "__pycache__", "results", "data", "figs"],
        "py_executable": "uv run",
    }
    
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
    )
    print(f"Job submitted: {job_id}")
    print(f"Dashboard: {address}/#/jobs/{job_id}")
    print("Streaming logs...\n" + "=" * 60)
    
    # Stream logs in real-time
    prev_log_len = 0
    while True:
        status = client.get_job_status(job_id)
        logs = client.get_job_logs(job_id)
        
        # Print new log lines
        if len(logs) > prev_log_len:
            sys.stdout.write(logs[prev_log_len:])
            sys.stdout.flush()
            prev_log_len = len(logs)
        
        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
            break
        
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"Job {job_id} finished with status: {status}")
    return job_id

def check_job(job_id: str, address: str = DEFAULT_DASHBOARD_URL):
    """Check status and print logs for a job."""
    client = get_client(address)
    status = client.get_job_status(job_id)
    print(f"Job {job_id} status: {status}")
    logs = client.get_job_logs(job_id)
    print("--- Logs (last 30 lines) ---")
    print('\n'.join(logs.split('\n')[-30:]))

def main():
    """CLI entrypoint."""
    import argparse
    parser = argparse.ArgumentParser(description="Ray + uv Utilities")
    parser.add_argument("--address", type=str, default=DEFAULT_DASHBOARD_URL,
                        help=f"Ray Dashboard URL (default: {DEFAULT_DASHBOARD_URL})")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Run command (synchronous submit + wait)
    run_parser = subparsers.add_parser("run", help="Submit and wait for a job (synchronous)")
    run_parser.add_argument("entrypoint", type=str, nargs="?", 
                           default="uv run simulate_results.py",
                           help="Command to run (default: 'uv run simulate_results.py')")
    
    # Check job command
    check_parser = subparsers.add_parser("check-job", help="Check job status and logs")
    check_parser.add_argument("job_id", type=str, help="Job ID to check")

    args = parser.parse_args()
    
    if args.command == "run":
        submit_and_wait(entrypoint=args.entrypoint, address=args.address)
    elif args.command == "check-job":
        check_job(args.job_id, address=args.address)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
