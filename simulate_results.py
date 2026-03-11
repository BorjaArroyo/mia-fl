import argparse
from src.simulator import TheorySimulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Theory Simulation on Ray")
    parser.add_argument("--cluster", type=str, default=None, 
                        help="Ray cluster address (e.g. 'ray://192.168.0.158:10001'). Default: None (local)")
    parser.add_argument("--limit", type=int, default=2000, help="Dataset limit per client loading")
    args = parser.parse_args()

    sim = TheorySimulator()
    sim.run(address=args.cluster, limit=args.limit)
