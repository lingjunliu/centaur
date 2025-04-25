import sys, time, os
import numpy as np
from .input_generators import get_random_input, get_abstract_input
from utils.api_utils import get_driver, get_signatures
from utils.misc import get_tmp_dir, create_subdir, get_dir_in_root, save_to_new_pkl

def random_fuzz(api, seed, duration, n_max=0, n_valid=0, lib="torch"):
    api_signature = get_signatures()[api]
    api_driver = get_driver(api, lib=lib)
    
    valid = 0
    invalid = 0
    abstract_inputs = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        rng = np.random.default_rng(seed)
        input_dict = get_random_input(api_signature, rng)
        try:            
            out_cpu = api_driver(input_dict, cpu=True)
        except:
            invalid += 1
        else:
            valid += 1
            # Save 
            if valid <= n_valid:
                abs_inp = get_abstract_input(input_dict, api_signature)
                # Save the abstract input along with the seed
                abstract_inputs.append((abs_inp, seed))
        
        print(f"Valid inputs for {api}: {valid}", end="\r", flush=True)
        
        # Check if maximum number of inputs reached
        if n_max > 0 and (valid+invalid) == n_max:
            break
        
        seed += 1
    
    return valid, invalid, abstract_inputs

def main():
    api = sys.argv[1] if len(sys.argv) > 1 else "argmin" # default api
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30   # seconds
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0    # maximum numberof inputs
    
    n_valid = 5 # number of inputs to save to infer invariants with
    seed = 42   # seed for reproduction
    tmp_results = create_subdir(get_tmp_dir(), "rand_results")
    csv_file = os.path.join(tmp_results, f"{api}_{duration}_{seed}.csv")
    
    print(f"Started fuzzing {api} for {duration} seconds...")
    valid, invalid, abstract_inputs = random_fuzz(api, seed, duration, n_max=n_max, n_valid=n_valid, lib="torch")
    valid_prcnt = round(valid*100/(valid+invalid),4) if (valid+invalid) > 0 else 0
    result = f"{api},{valid},{invalid},{valid_prcnt}\n"
    print(f"\nValid: {valid} | Invalid: {invalid} | Validity Percentage: {valid_prcnt}")
    
    with open(csv_file, "w") as f:
        f.write(result)
        
    # Save valid inputs for invariant inference
    pkl_file = os.path.join(get_dir_in_root("valid_inputs"), f"{api}.pkl")
    save_to_new_pkl(pkl_file, abstract_inputs)

if __name__ == "__main__":
    main()