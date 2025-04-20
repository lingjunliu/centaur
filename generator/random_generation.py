import sys, time, os
import numpy as np
from .input_generators import gen_ran_ll, gen_concrete_input
from utils.api_utils import get_driver, get_signatures
from utils.misc import get_tmp_dir, create_subdir

def random_fuzz(api, seed, duration, lib="torch"):
    api_signature = get_signatures()[api]
    api_driver = get_driver(api, lib=lib)
    rng = np.random.default_rng(seed)
    
    valid = 0
    invalid = 0
    
    start_time = time.time()
    while time.time() - start_time < duration:
        input_dict = {}
        for arg, domain in api_signature.items():
            # TODO: Add support for tensor_list
            if domain == "tensor_list":
                domain = "tensor"   # hack until tensor_list is supported
            
            ll = gen_ran_ll(domain, rng)    # get abstract form            
            input_dict[arg] = gen_concrete_input(domain, ll, rng) # concretize
        try:            
            out_cpu = api_driver(input_dict, cpu=True)
        except:
            invalid += 1
        else:
            valid += 1
        
        print(f"Valid inputs for {api}: {valid}", end="\r", flush=True)
    
    return valid, invalid

def main():
    api = sys.argv[1] if len(sys.argv) > 1 else "argmin" # default api
    seed = 42   # seed for reproduction
    duration = 30   # seconds
    tmp_results = create_subdir(get_tmp_dir(), "results")
    csv_file = os.path.join(tmp_results, f"{api}_{duration}_{seed}.csv")
    
    print(f"Started fuzzing {api} for {duration} seconds...")
    valid, invalid = random_fuzz(api, seed, duration, lib="torch")
    valid_prcnt = round(valid*100/(valid+invalid),4) if (valid+invalid) > 0 else 0
    result = f"{api},{valid},{invalid},{valid_prcnt}\n"
    print(f"\nValid: {valid} | Invalid: {invalid} | Validity Percentage: {valid_prcnt}")
    
    with open(csv_file, "w") as f:
        f.write(result)

if __name__ == "__main__":
    main()