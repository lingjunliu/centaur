
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def max_pool1d_inputs():
    list_of_inputs = []

    input1 = np.random.rand(1, 5, 3).astype(np.float32)
    kernel_size1 = 2
    stride1 = 1
    padding1 = 0
    dilation1 = 1
    return_indices1 = True
    ceil_mode1 = False
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "return_indices": return_indices1,
        "ceil_mode": ceil_mode1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool1d"] = max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool1d'.")


check_valid('torch.nn.functional.max_pool1d', generated_inputs['torch.nn.functional.max_pool1d'], lib="torch", suffix=0)
