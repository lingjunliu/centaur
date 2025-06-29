import sys
from .harness_z3 import run_api_with_duration as fuzz_with_z3
from datetime import datetime

def run_fuzz(api, duration, n_max, lib, seed, print_details, use_reference):
    mode = "z3" # default mode, optimizer partially implemented

    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"


    # Logging run config at the beginning
    print(f"Fuzzing with the {api} driver on {lib}. Mode: {mode}, seed: {seed}.")
    print('Started fuzzing at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if mode.strip().lower() == "z3":
        fuzz_with_z3(api, duration, n_max=n_max, seed=seed, lib=lib, print_details=print_details, use_reference=use_reference)
    else:
        raise Exception(f"Unsupported mode: {mode}. Supported mode: z3.")
    

def main():
    if len(sys.argv) < 3:
        print("Usage: python fuzz.py <api> <duration> <n_max, optional> <lib, default: torch> <seed, optional> <print_details, optional>")
        return
    
    api = sys.argv[1]
    duration = int(sys.argv[2])
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    print_details = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
    use_reference = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
    
    run_fuzz(api, duration, n_max, lib, seed, print_details, use_reference)

if __name__ == "__main__":
    main()