
import re, sys, shutil, os, pickle
import numpy as np
from generator.input_generators import concretize_input, abstract_print
from utils.misc import map_torch_to_driver, create_subdir, get_dir_in_root, get_tmp_dir, read_pkl
from utils.api_utils import get_driver, get_signatures
from .oracle import oracle_crash

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def replace_function_invocation(script, old_function_name, new_function_name):
    pattern = rf"(?:^|\s|=){old_function_name}\((.*?)\)"
    regex = re.compile(pattern, re.DOTALL)
    matches = re.finditer(regex, script)
    for match in matches:
        old_args = match.group(1)
        new_invocation = f" {new_function_name}({old_function_name}, {old_args})"
        script = script.replace(match.group(), new_invocation)
    
    return script

def prefix():
    return """# Monkey Patched Driver
import pickle, os

def wrapper_1(func, *args, **kwargs):
    filename = os.path.abspath(__file__)[:-3] + '_1.pkl'
    with open(filename, 'rb') as f:
        input_list = pickle.load(f)
        input_dict = {
            'args': args,
            'kwargs': kwargs
        }
        input_list.append(input_dict)
    with open(filename, 'wb') as f:
        pickle.dump(input_list, f)
    return func(*args, **kwargs)

def wrapper_2(func, *args, **kwargs):
    filename = os.path.abspath(__file__)[:-3] + '_2.pkl'
    with open(filename, 'rb') as f:
        input_list = pickle.load(f)
        input_dict = {
            'args': args,
            'kwargs': kwargs
        }
        input_list.append(input_dict)
    with open(filename, 'wb') as f:
        pickle.dump(input_list, f)
    return func(*args, **kwargs)

"""

def generate_driver(filename_1, filename_2, torch_api):
    return f"""# Driver to run all the inputs
import pickle, torch

filename_1 = '{filename_1}'
filename_2 = '{filename_2}'
with open(filename_1, 'rb') as f:
    input_list_1 = pickle.load(f)
with open(filename_2, 'rb') as f:
    input_list_2 = pickle.load(f)

print(len(input_list_1), len(input_list_2))
ran = 0

if len(input_list_1) != len(input_list_2):
    for input_dict in input_list_1:
        try:
            output = {torch_api}(*input_dict['args'], **input_dict['kwargs'])
            ran += 1
        except Exception as e:
            print(e.__class__.__name__ + ": " + str(e))
else:
    for input_dict_1, input_dict_2 in zip(input_list_1, input_list_2):
        try:
            output = {torch_api}(*input_dict_1['args'], **input_dict_1['kwargs'])(*input_dict_2['args'], **input_dict_2['kwargs'])
            ran += 1
        except Exception as e:
            print(e.__class__.__name__ + ": " + str(e))
print(ran)
"""

def main():
    driver = sys.argv[1]
    n_inputs = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    print_details = True
    lib = "torch"
    seed = 19
    rng_choice = np.random.default_rng(seed)

    patch_dir = create_subdir(CUR_DIR, "patched_drivers")     # directory to save patched drivers and inputs
    src_driver = os.path.join(get_dir_in_root("drivers"), f"{driver}.py")  # path to source driver
    patched_driver = os.path.join(patch_dir, f"{driver}.py")        # path to save patched driver
    shutil.copy(src_driver, patched_driver)

    validity_dir = create_subdir(get_tmp_dir(), "validity_results")  # directory to save validity results
    valdity_csv = os.path.join(validity_dir, f"{driver}.csv")  # path to save validity results

    torch_to_driver, driver_to_torch = map_torch_to_driver()
    torch_api = driver_to_torch[driver]
    # patch the driver
    with open(patched_driver, "r") as f:
        code = f.read()

        # Save arg_class in case this is a functional class
        arg_class = None
        for line in code.split("\n"):
            if f"{torch_api}(" in line:
                tokens = line.split("=")
                if len(tokens) > 1:
                    arg_class = tokens[0].strip()
                    break
        
        code = f"{prefix()}\n{code}"
        code = replace_function_invocation(code, torch_api, "wrapper_1")

        print(f"arg_class: {arg_class}")

        if arg_class:
            code = replace_function_invocation(code, arg_class, "wrapper_2")
    
    # Save the patched driver
    with open(patched_driver, "w") as f:
        f.write(code)
    
    # Create two pickle files with empty lists
    filename_1 = os.path.join(patch_dir, f"{driver}_1.pkl")
    filename_2 = os.path.join(patch_dir, f"{driver}_2.pkl")
    with open(filename_1, "wb") as f:
        pickle.dump([], f)
    with open(filename_2, "wb") as f:
        pickle.dump([], f)

    # Create the driver to run all the inputs
    driver_code = generate_driver(filename_1, filename_2, torch_api)
    driver_file = os.path.join(patch_dir, f"{driver}_cov_in_loop.py")
    with open(driver_file, "w") as f:
        f.write(driver_code)

    input_dir = os.path.join(get_tmp_dir(), "fuzz_inputs")
    input_file = os.path.join(input_dir, f"{driver}_{lib}_inputs.pkl")
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        input_file = os.path.join(input_dir, f"{driver}_inputs.pkl")
        print(f"Trying to load {input_file} instead.")
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist either.")
            return
    
    signature = get_signatures()[driver]
    generated_inputs = read_pkl(input_file)
    driver_function = get_driver(driver, lib=lib, module="eval.patched_drivers")
    valid = 0
    invalid = 0
    crash = 0
    excp = 0
    total = 0
    valid_prcnt = 0

    if n_inputs > 0:
        # Choose n_inputs inputs randomly
        indices = list(range(0, len(generated_inputs)))
        rng_choice.shuffle(indices)
        generated_inputs = [generated_inputs[i] for i in indices[:n_inputs]]

    print(f"Running patched driver for {len(generated_inputs)} inputs.")
    for best_distance, abs_input, seed in generated_inputs:
        rng = np.random.default_rng(seed)
        # Get the input dictionary
        input_dict = concretize_input(abs_input, signature, rng)
        status, exception_message = oracle_crash(driver_function, input_dict, cpu=True)
        if status == "invalid":
            invalid += 1
        elif status == "cpu_excp":
            excp += 1
        elif status == "cpu_crash":
            crash += 1
        else:
            valid += 1
        
        if print_details:
            print(f"Input:\n{abstract_print(abs_input, signature)}")
            print(f"Status: {status}")
            if exception_message > "":
                print(f"Exception: {exception_message}")
        
        total += 1
        valid_prcnt = ((total-invalid) / total) * 100 if total > 0 else 0
        with open(valdity_csv, "w") as f:
            # api,valid,invalid,crash,exception,total,valid_prcnt        
            f.write(f"{driver},{valid},{invalid},{crash},{excp},{total},{valid_prcnt}\n")

    print(f"Total inputs: {total} | Valid Percentage: {valid_prcnt:.2f}%")

if __name__ == "__main__":
    main()