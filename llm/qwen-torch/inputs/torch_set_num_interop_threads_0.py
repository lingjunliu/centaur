
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def set_num_interop_threads_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = 1   # integer
    input_dict = {
        "num_threads": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_num_interop_threads"] = set_num_interop_threads_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_num_interop_threads' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_num_interop_threads'.")


check_valid('torch.set_num_interop_threads', generated_inputs['torch.set_num_interop_threads'], lib="torch", suffix=0)
