import sys, time, os, logging
import numpy as np
from .input_generators import get_random_input, abstract_print
from utils.new_api_utils import get_signature, get_n_variations, get_lib_version
from utils.misc import get_tmp_dir, create_subdir, get_dir_in_root, save_to_pkl, save_to_new_pkl
from eval.oracle import oracle_crash

logger = logging.getLogger(__name__)

def random_fuzz(api, seed, duration, n_max=0, n_valid=0, lib="torch", logfile=None):
    if not logfile:
        logfile = f"logs/{api}_ran_excp.log"
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,                                     # Minimum log level
        format='%(asctime)s | %(levelname)s | %(message)s',     # Log format
        filename=logfile,                                       # Log file path
        filemode="w"                                            # Append/Write mode
    )

    n_variants = get_n_variations(api, lib=lib)
    
    valid = 0
    invalid = 0
    crash = 0
    abstract_inputs = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        rng = np.random.default_rng(seed)
        suffix = 0 if n_variants == 1 else rng.integers(1, n_variants+1)
        api_signature = get_signature(api, lib=lib, suffix=suffix)
        input_dict, abs_inp = get_random_input(api_signature, rng, lib=lib)
        status, exception_message, traceback = oracle_crash(api, input_dict, cpu=True, lib=lib, include_traceback=True)
        if len(exception_message.splitlines()) > 1:
            exception_message = ' '.join(exception_message.splitlines())
        if status == "nominal":
            valid += 1
            # Save 
            if valid <= n_valid:
                # Save the abstract input along with the seed
                abstract_inputs.append((abs_inp, seed, suffix))
        elif status == "invalid":
            invalid += 1
            # Traceback for debugging
            logger.error(exception_message)
            logger.info(traceback)
            logger.info(f"Status: {status}, Seed: {seed}, Suffix: {suffix}")
            logger.info(f"Abstract Input:\n{abstract_print(abs_inp, api_signature)}")
        elif status == "cpu_crash":
            crash += 1
            # Traceback for debugging
            logger.error(exception_message)
            logger.info(traceback)
            logger.info(f"Status: {status}, Seed: {seed}, Suffix: {suffix}")
            logger.info(f"Abstract Input:\n{abstract_print(abs_inp, api_signature)}")
        
        print(f"Valid: {valid} | Invalid: {invalid} | Crash: {crash}", end="\r", flush=True)
        
        # Check if maximum number of inputs reached
        if n_max > 0 and (valid+invalid) == n_max:
            break
        
        seed += 1
    
    return valid, invalid, crash, abstract_inputs

def main():
    api = sys.argv[1] if len(sys.argv) > 1 else "argmin" # default api
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30   # seconds
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0    # maximum numberof inputs
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"  # library, default is torch

    api = get_lib_version(api, lib=lib)
    
    override_valid = True # setting to True will override the saved valid inputs
    n_valid = 5 # number of inputs to save to infer invariants with
    seed = 42   # seed for reproduction
    tmp_results = create_subdir(get_tmp_dir(), "rand_results")
    csv_file = os.path.join(tmp_results, f"{api}_{duration}_{seed}.csv")
    logfile = os.path.join(tmp_results, f"{api}_excp.log")
    
    print(f"Started fuzzing {api} for {duration} seconds...")
    valid, invalid, crash, abstract_inputs = random_fuzz(api, seed, duration, n_max=n_max, n_valid=n_valid, lib=lib, logfile=logfile)
    total = valid + invalid + crash
    valid_prcnt = round((valid+crash)*100/total,2) if total > 0 else 0
    result = f"{api},{valid},{invalid},{crash},{total},{valid_prcnt}\n"
    print(f"\nValid: {valid} | Invalid: {invalid} | Crash: {crash} | Validity Percentage: {valid_prcnt}")
    
    with open(csv_file, "w") as f:
        f.write(result)
        
    # Save valid inputs for invariant inference
    pkl_file = os.path.join(get_dir_in_root(f"valid_inputs_{lib}"), f"{api}.pkl")
    if override_valid:
        save_to_pkl(pkl_file, abstract_inputs)
    else:
        save_to_new_pkl(pkl_file, abstract_inputs)

if __name__ == "__main__":
    main()