from utils.proc import run_with_timeout
from utils.api_utils import get_driver, get_signatures
from utils.misc import get_tmp_dir, create_subdir, read_pkl, save_to_pkl, get_input_size
from generator.input_generators import abstract_print, concretize_input
from copy import deepcopy
import numpy as np
import sys, os
import time

def max_diff_with_indices(a, b, rtol=1e-7, atol=0.01, equal_nan=True, equal_inf=True):
    """
        Returns the max diff, the indices of the max diff, and the values of a and b at those indices.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # If input is boolean, cast to int for safe arithmetic
    if a.dtype == bool:
        a = a.astype(int)
    if b.dtype == bool:
        b = b.astype(int)

    # Base isclose comparison (only valid on float/int)
    is_close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)

    if equal_nan:
        is_close |= (np.isnan(a) & np.isnan(b))
    if equal_inf:
        is_close |= ((a == np.inf) & (b == np.inf)) | ((a == -np.inf) & (b == -np.inf))

    mismatch_mask = ~is_close

    if not np.any(mismatch_mask):
        return 0.0, (), None, None

    diffs = np.abs(a[mismatch_mask] - b[mismatch_mask])
    max_diff = np.nanmax(diffs)

    mismatch_diffs = np.abs(a - b)
    max_locs = np.argwhere(mismatch_mask & (mismatch_diffs == max_diff))
    indices = [tuple(idx) for idx in max_locs]
    a_vals = [a[idx] for idx in indices]
    b_vals = [b[idx] for idx in indices]

    return max_diff, indices, a_vals, b_vals

def check_crash(return_code, exception_message):
    """
    Check if the exception message indicates a crash.
    
    Args:
        exception_message (str): The exception message to check.
    
    Returns:
        bool: True if the exception message indicates a crash, False otherwise.
    """
    list_of_strings = [
        "Segmentation fault",
        "Aborted",
        "Illegal instruction",
        "Floating point exception",
        "Bus error",
        "Killed",
        "Abort trap",
        "Process killed",
        "MemoryError",
        "INTERNAL ASSERT ERROR",
        "please report a bug",
        "CUDA out of memory",
        "CUDA error"
        "Timeout"
        # Add more crash-related strings as needed
    ]
    
    if return_code == 0:
        return False
    elif return_code < 0:
        return True
    else:
        for string in list_of_strings:
            if string.lower() in exception_message.lower():
                return True
    
    return False

def compare_two(elem1, elem2, rtol=1e-07, atol=0.01):
    indices = None
    elem1_val = None
    elem2_val = None
    if type(elem1) != type(elem2):
        return False, None, indices, elem1_val, elem2_val
    
    if isinstance(elem1, list):
        elem1 = np.array(elem1).flatten()
        elem2 = np.array(elem2).flatten()
    
    if elem1.size > 0 and elem2.size > 0:
        max_diff, indices, elem1_val, elem2_val = max_diff_with_indices(elem1, elem2, rtol=rtol, atol=atol, equal_nan=True, equal_inf=True)
    elif elem1.size == 0 and elem2.size == 0:
        max_diff = 0.0
    else:
        max_diff = None
        
    if isinstance(elem1, np.ndarray):
        if elem1.size != elem2.size:
            return False, max_diff, indices, elem1_val, elem2_val
    
    try:
        matched = np.allclose(elem1, elem2, rtol=rtol, atol=atol, equal_nan=True)
    except: # most likely not numeric
        matched = elem1 == elem2

    return matched, max_diff, indices, elem1_val, elem2_val

def consistent(output1, output2, rtol=1e-07, atol=1e-08):
    matched = True
    indices = None
    elem1_val = None
    elem2_val = None

    if type(output1) != type(output2):
        print(f"Expected two dicts, got {type(output1)} and {type(output2)}")
        return False, None, indices, elem1_val, elem2_val

    for name in output1.keys():
        matched, max_diff, indices, elem1_val, elem2_val = compare_two(output1[name], output2[name], rtol=rtol, atol=atol)

        if matched == False:
            return False, max_diff, indices, elem1_val, elem2_val

    return matched, max_diff, indices, elem1_val, elem2_val

def oracle_crash(driver, input_dict, timeout=10, cpu=True):
    """
        Check if the input is valid for the given API with a timeout or
        if it crashes. This runs on either CPU or GPU.
        
        Args:
            driver (func): The API driver function to run.
            input_dict (dict): The input dictionary to check.
            timeout (int): The timeout in seconds.
        
        Returns:
            tuple: (status, exception_message)
            - ("nominal", "") if the input is valid and nominal.
            - ("invalid", exception_message) if the input is invalid.
            - ("cpu_crash", exception_message) if the API crashes raising a signal on CPU.
            - ("gpu_crash", exception_message) if the API crashes raising a signal on GPU.
            - ("cpu_excp", exception_message) if the API throws an exception on CPU.
            - ("gpu_excp", exception_message) if the API throws an exception on GPU.
    """
    return_code, output, exception_message = run_with_timeout(driver, timeout, input_dict, cpu=cpu)
    
    if return_code < 0: # signal raised
        return ("cpu_crash", exception_message) if cpu else ("gpu_crash", exception_message)
    elif check_crash(return_code, exception_message): # non-signal error
        return ("cpu_excp", exception_message) if cpu else ("gpu_excp", exception_message)
    elif return_code > 0:
        return ("invalid", exception_message)
    else:
        return ("nominal", "")

def oracle_diff(driver, signature, input_dict, timeout=10, atol=1e-08, detailed=True):
    """
    Run the API with a timeout on cpu and gpu, and compare the outputs.
    
    Args:
        driver (func): The API driver function to run.
        signature (dict): The API signature to use.
        input_dict (dict): The input dictionary to pass to the API.
        timeout (int): The timeout in seconds.
        detailed (bool): Whether to return detailed information about the differences (i.e. all indices where we get max diff).
    
    Returns:
        tuple: A tuple containing the oracle result and the exception message (if any).
    
    Exceptions:
        - If the API execution crashes on cpu, returns ("cpu_crash", exception_message_cpu).
        - If the API execution crashes on gpu, returns ("gpu_crash", exception_message_gpu).
        - If the API execution raises an exception on cpu, returns ("cpu_excp", exception_message_cpu).
        - If the API execution raises an exception on gpu, returns ("gpu_excp", exception_message_gpu).
        - If the outputs are inconsistent, returns ("inconsistent", max_diff).
        - If the execution faced exception on both cpu and gpu, returns ("invalid", exception_message_cpu, exception_message_gpu).
        - If the execution is invalid on cpu, returns ("cpu_only_excp", exception_message_cpu).
        - If the execution is invalid on gpu, returns ("gpu_only_excp", exception_message_gpu).
        - If the execution is nominal, returns ("nominal", "").
    """
    # cpu
    return_code_cpu, output_cpu, exception_message_cpu = run_with_timeout(driver, timeout, deepcopy(input_dict), cpu=True)
    
    # check if the CPU execution crashed
    if return_code_cpu < 0: # signal raised
        return ("cpu_crash", exception_message_cpu)
    elif check_crash(return_code_cpu, exception_message_cpu):
        return ("cpu_excp", exception_message_cpu)
    
    # gpu
    return_code_gpu, output_gpu, exception_message_gpu = run_with_timeout(driver, timeout, deepcopy(input_dict), cpu=False)
    
    # check if the GPU execution crashed
    if return_code_gpu < 0: # signal raised
        return ("gpu_crash", exception_message_gpu)
    elif check_crash(return_code_gpu, exception_message_gpu):
        return ("gpu_excp", exception_message_gpu)
    
    # check if the outputs are the same
    if return_code_cpu != return_code_gpu:
        return ("cpu_only_excp", exception_message_cpu) if return_code_cpu != 0 else ("gpu_only_excp", exception_message_gpu)
    else:
        if return_code_cpu != 0:
            return ("invalid", exception_message_cpu, exception_message_gpu)
        const, max_diff, indices, elem1_val, elem2_val = consistent(output_cpu, output_gpu, atol=atol)
        if not const:
            try:
                return ("inconsistent", max_diff, indices[0], elem1_val[0], elem2_val[0]) if not detailed else ("inconsistent", max_diff, indices, elem1_val, elem2_val)
            except:
                return ("inconsistent", max_diff, indices, elem1_val, elem2_val)
    
    return ("nominal", "")

def save_state_oracle(api, result_summary, oracle_results):
    # Save the results
    results_dir = create_subdir(get_tmp_dir(), "oracle_results")
    save_to_pkl(os.path.join(results_dir, f"{api}.pkl"), oracle_results)
    csv_file = os.path.join(results_dir, f"{api}.csv")
    with open(csv_file, "w") as f:
        # api,nominal,invalid,cpu_crash,gpu_crash,cpu_excp,gpu_excp,cpu_only_excp,gpu_only_excp,inconsistent,max_diff
        f.write(f"{api}," + ",".join([str(x) for x in result_summary.values()]) + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python oracle.py <api_name>")
        return
    
    A_TOL = 1e-02   # Tolerance: 0.01 (from FreeFuzz)
    TIMEOUT = 10    # seconds
    print_details = True
    saving_interval = 100 # inputs
    
    api = sys.argv[1]
    # Optional: low and high values for input generation [low, high)
    low = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    high = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    # Directory containing the input files
    tmp = get_tmp_dir()
    input_dir = os.path.join(tmp, "fuzz_inputs")
    input_file = os.path.join(input_dir, f"{api}_inputs.pkl")
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return
    
    generated_inputs = read_pkl(input_file)
    signature = get_signatures()[api]
    driver = get_driver(api)
    
    if low != -1 and high != -1:
        print(f"Filtering generated inputs from {low} to {high}")
        # Filter the generated inputs based on the low and high values
        if low < len(generated_inputs):
            generated_inputs = generated_inputs[low:min(high, len(generated_inputs))]
        else:
            print(f"Low value {low} is out of range for the generated inputs.")
            return
    elif low != -1:
        print(f"Filtering generated inputs from {low} to the end")
        # Filter the generated inputs based on the low value
        if low < len(generated_inputs):
            generated_inputs = generated_inputs[low:]
        else:
            print(f"Low value {low} is out of range for the generated inputs.")
            return
    
    oracle_results = []
    result_summary = {
        "nominal": 0,
        "invalid": 0,
        "cpu_crash": 0,
        "gpu_crash": 0,
        "cpu_excp": 0,
        "gpu_excp": 0,
        "cpu_only_excp": 0,
        "gpu_only_excp": 0,
        "inconsistent": 0,
        "max_diff": 0
    }
    
    total = len(generated_inputs)
    total_time = 0
    i = 0
    
    for best_distance, abs_input, seed in generated_inputs:
        i += 1
        if i % saving_interval == 0:
            save_state_oracle(api, result_summary, oracle_results)

        rng = np.random.default_rng(seed)
        # Get the input dictionary
        input_dict = concretize_input(abs_input, signature, rng)
        
        # Run the oracle
        start_time = time.time()
        result_tuple = oracle_diff(driver, signature, input_dict, timeout=TIMEOUT, atol=A_TOL)
        duration = round(time.time() - start_time, 2)
        total_time += duration
        oracle_results.append(result_tuple)
        result_summary[result_tuple[0]] += 1
        if result_tuple[0] == "inconsistent":
            result_summary["max_diff"] = max(result_summary["max_diff"], result_tuple[1])
        print(f"Checked {i}/{total} inputs | Took {duration} s | Avg: {round(total_time/i, 2)} s | Size: {round(get_input_size(abs_input, signature), 2)} MB           ", end="\r", flush=True)
        
        if result_tuple[0] not in ["nominal", "invalid"] and print_details:
            print(' | '.join([str(x) for x in result_tuple]))
            print('Seed:', seed)
            print(abstract_print(abs_input, signature))

    # Print the summary
    print(f"Oracle results for {api}:")
    for result, count in result_summary.items():
        print(f"{result}: {count}")
        
    save_state_oracle(api, result_summary, oracle_results)

if __name__ == "__main__":
    main()