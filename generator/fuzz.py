import sys
from .harness import run_api_with_duration as fuzz_with_optimizer
from .harness_z3 import run_api_with_duration as fuzz_with_z3

def main():
    if len(sys.argv) < 3:
        print("Usage: python fuzz.py <api> <duration> <mode, optional (default: z3)> <n_max, optional>")
        return
    
    api = sys.argv[1]
    duration = int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "z3"
    n_max = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    
    if mode.strip().lower() == "z3":
        fuzz_with_z3(api, duration, n_max=n_max, limit=limit)
    else:
        fuzz_with_optimizer(api, duration, n_max, limit)

if __name__ == "__main__":
    main()