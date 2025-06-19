import sys, os, pickle
import numpy as np
from generator.input_generators import concretize_input, abstract_print
from utils.misc import create_subdir, get_tmp_dir, read_pkl
from utils.new_api_utils import get_signature, get_lib_version, get_input

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_driver_torch(dir, torch_api):
    return f"""# Driver to run all the inputs
import pickle, os, sys

files = os.listdir('{dir}')
logfile = os.path.join('{dir}', 'log.txt')
with open(logfile, 'w') as log:
    log.write('Starting driver for {torch_api}\\n')
input_list = []
for file in files:
    if file.endswith('.pkl'):
        input_list.append(os.path.join('{dir}', file))

print('Input files found:', len(input_list))

if len(input_list) == 0:
    print('No input files found in the directory.')
    sys.exit(0)

ran = 0
excp = 0

import torch
for file in input_list:
    with open(file, 'rb') as f:
        input_dict = pickle.load(f)
    try:
        if 'inner' in input_dict.keys() and len(input_dict['inner'].keys()) > 0:
            output = {torch_api}(*input_dict['args'], **input_dict['kwargs'])(*input_dict['inner']['args'], **input_dict['inner']['kwargs'])
        else:
            output = {torch_api}(*input_dict['args'], **input_dict['kwargs'])
        ran += 1
        with open(logfile, 'a') as log:
            log.write('Ran: ' + str(ran) + '\\n')
    except Exception as e:
        print(e.__class__.__name__ + ": " + str(e))
        excp += 1
        with open(logfile, 'a') as log:
            log.write('Excp: ' + str(excp) + '\\n')
            log.write(str(e.__class__.__name__) + ": " + str(e) + '\\n')

print('Ran:', ran)
print('Exception:', excp)
"""

def main():
    api = sys.argv[1]
    n_inputs = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    print_details = True
    lib = "torch"
    cur_seed = 19
    rng_choice = np.random.default_rng(cur_seed)

    api = get_lib_version(api, lib=lib)

    patch_dir_root = create_subdir(CUR_DIR, "patched_drivers")     # directory to save patched drivers and inputs
    patch_dir = create_subdir(patch_dir_root, api)                 # subdirectory for api
    
    # Create the driver to run all the inputs
    driver_code = generate_driver_torch(patch_dir, api)
    driver_file = os.path.join(patch_dir, f"{api}_cov_in_loop.py")
    with open(driver_file, "w") as f:
        f.write(driver_code)

    input_dir = os.path.join(get_tmp_dir(), "fuzz_inputs")
    input_file = os.path.join(input_dir, f"{api}_{lib}_inputs.pkl")
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        input_file = os.path.join(input_dir, f"{api}_inputs.pkl")
        print(f"Trying to load {input_file} instead.")
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist either.")
            return
    
    generated_inputs = read_pkl(input_file)
    
    total = 0

    if n_inputs > 0:
        # Choose n_inputs inputs randomly
        indices = list(range(0, len(generated_inputs)))
        rng_choice.shuffle(indices)
        generated_inputs = [generated_inputs[i] for i in indices[:n_inputs]]

    print(f"Running patched driver for {len(generated_inputs)} inputs.")
    for best_distance, abs_input, seed, suffix in generated_inputs:
        signature = get_signature(api, lib=lib, suffix=suffix)
        rng = np.random.default_rng(seed)
        # Get the input dictionary
        input_dict = concretize_input(abs_input, signature, rng)
        if print_details:
            print(f"Input:\n{abstract_print(abs_input, signature)}")
        
        lib_input = get_input(api, input_dict, cpu=True, lib=lib)
        total += 1

        # Save the input to a file
        input_file = os.path.join(patch_dir, f"input_{total}.pkl")
        with open(input_file, "wb") as f:
            pickle.dump(lib_input, f)

    print(f"Total inputs: {total}")

if __name__ == "__main__":
    main()