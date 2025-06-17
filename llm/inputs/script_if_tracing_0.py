
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def script_if_tracing_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean condition
    input_dict = {
        "condition": np.array(True, dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Boolean condition with numpy False
    input_dict = {
        "condition": np.array(False, dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean condition as an integer (0 or 1)
    input_dict = {
        "condition": np.array(1, dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean condition as an integer zero
    input_dict = {
        "condition": np.array(0, dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean condition based on a comparison
    input_dict = {
        "condition": np.array(5) > np.array(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.jit.script_if_tracing"] = script_if_tracing_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.script_if_tracing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.script_if_tracing'.")

check_valid('torch.jit.script_if_tracing', generated_inputs['torch.jit.script_if_tracing'], lib="torch")
