import sys
from .harness import run_api_with_duration as fuzz_with_optimizer
from .harness_z3 import run_api_with_duration as fuzz_with_z3
from datetime import datetime

def main():
    if len(sys.argv) < 3:
        print("Usage: python fuzz.py <api> <duration> <mode, optional (default: z3)> <n_max, optional>")
        return
    
    api = sys.argv[1]
    duration = int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "z3"
    n_max = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 200
    lib = sys.argv[7] if len(sys.argv) > 7 else "torch"
    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"
        
    # Logging run config at the beginning
    print(f"Fuzzing with the {api} driver on {lib}. Mode: {mode}, seed: {seed}.")
    print('Started fuzzing at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if mode.strip().lower() == "z3":
        # split duration between model generation and fuzzing
        # use <limit> as the ratio
        model_gen_duration = int(duration*limit/100)
        fuzz_duration = duration - model_gen_duration
        max_model = n_max
        fuzz_with_z3(api, model_gen_duration, fuzz_duration, max_model, n_max=n_max, seed=seed, lib=lib)
    else:
        fuzz_with_optimizer(api, duration, n_max, limit, lib=lib)

if __name__ == "__main__":
    main()