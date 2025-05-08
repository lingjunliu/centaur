from utils.proc import run_with_timeout
from utils.api_utils import get_driver, get_signatures
from utils.misc import get_tmp_dir, create_subdir, read_pkl, save_to_new_pkl
from generator.input_generators import get_abstract_input, concretize_input
from copy import deepcopy
import numpy as np
import sys, os

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
        "CUDA error",
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

def compare_two(elem1, elem2, rtol=1e-05, atol=1e-08):
    if type(elem1) != type(elem2):
        return False
    
    if isinstance(elem1, list):
        elem1 = np.array(elem1).flatten()
        elem2 = np.array(elem2).flatten()
        
    if isinstance(elem1, np.ndarray):
        if len(elem1) != len(elem2):
            return False
    
    try:
        matched = np.allclose(elem1, elem2, rtol=rtol, atol=atol, equal_nan=True)
    except: # most likely not numeric
        matched = elem1 == elem2

    return matched

def consistent(output1, output2, rtol=1e-05, atol=1e-08):
    matched = True

    if type(output1) != type(output2):
        print(f"Expected two dicts, got {type(output1)} and {type(output2)}")
        return False

    for name in output1.keys():
        matched = compare_two(output1[name], output2[name], rtol=rtol, atol=atol)

        if matched == False:
            return False

    return matched

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

def oracle_diff(driver, signature, input_dict, timeout=10, atol=1e-08):
    """
    Run the API with a timeout on cpu and gpu, and compare the outputs.
    
    Args:
        driver (func): The API driver function to run.
        signature (dict): The API signature to use.
        input_dict (dict): The input dictionary to pass to the API.
        timeout (int): The timeout in seconds.
    
    Returns:
        tuple: A tuple containing the oracle result and the exception message (if any).
    
    Exceptions:
        - If the API execution crashes on cpu, returns ("cpu_crash", exception_message_cpu).
        - If the API execution crashes on gpu, returns ("gpu_crash", exception_message_gpu).
        - If the outputs are inconsistent, returns ("inconsistent", abstract_input).
        - If the execution faced exception on both cpu and gpu, returns ("invalid", exception_message_cpu, exception_message_gpu).
        - If the execution is nominal, returns ("nominal", "").
    """
    # cpu
    return_code_cpu, output_cpu, exception_message_cpu = run_with_timeout(driver, timeout, deepcopy(input_dict), cpu=True)
    
    # check if the CPU execution crashed
    if check_crash(return_code_cpu, exception_message_cpu):
        return ("cpu_crash", exception_message_cpu)
    
    # gpu
    return_code_gpu, output_gpu, exception_message_gpu = run_with_timeout(driver, timeout, deepcopy(input_dict), cpu=False)
    
    # check if the GPU execution crashed
    if check_crash(return_code_gpu, exception_message_gpu):
        return ("gpu_crash", exception_message_gpu)
    
    # check if the outputs are the same
    if return_code_cpu != return_code_gpu:
        return ("cpu_crash", exception_message_cpu) if return_code_cpu != 0 else ("gpu_crash", exception_message_gpu)
    else:
        if return_code_cpu != 0:
            return ("invalid", exception_message_cpu, exception_message_gpu)
        elif not consistent(output_cpu, output_gpu, atol=atol):
            return ("inconsistent", get_abstract_input(input_dict, signature))
    
    return ("nominal", "")

def main():
    if len(sys.argv) < 2:
        print("Usage: python oracle.py <api_name>")
        return
    
    A_TOL = 1e-02   # Tolerance: 0.01 (from FreeFuzz)
    TIMEOUT = 10    # seconds
    
    api = sys.argv[1]
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
    
    oracle_results = []
    result_summary = {
        "nominal": 0,
        "invalid": 0,
        "cpu_crash": 0,
        "gpu_crash": 0,
        "inconsistent": 0
    }
    
    total = len(generated_inputs)
    
    for best_distance, abs_input, seed in generated_inputs:
        rng = np.random.default_rng(seed)
        # Get the input dictionary
        input_dict = concretize_input(abs_input, signature, rng)
        
        # Run the oracle
        result_tuple = oracle_diff(driver, signature, input_dict, timeout=TIMEOUT, atol=A_TOL)
        oracle_results.append(result_tuple)
        result_summary[result_tuple[0]] += 1
        print(f"Checked {sum(result_summary.values())}/{total} inputs", end="\r", flush=True)

    # Print the summary
    print(f"Oracle results for {api}:")
    for result, count in result_summary.items():
        print(f"{result}: {count}")
        
    # Save the results
    results_dir = create_subdir(get_tmp_dir(), "oracle_results")
    save_to_new_pkl(os.path.join(results_dir, f"{api}.pkl"), oracle_results)
    csv_file = os.path.join(results_dir, f"{api}.csv")
    with open(csv_file, "w") as f:
        # api,nominal,invalid,cpu_crash,gpu_crash,inconsistent
        f.write(f"{api}," + ",".join([str(x) for x in result_summary.values()]) + "\n")

if __name__ == "__main__":
    main()