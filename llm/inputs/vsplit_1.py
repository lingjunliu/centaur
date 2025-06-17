
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vsplit_inputs():
    list_of_inputs = []

    # Case 1: Evenly divisible integer split
    t = np.arange(16.0).reshape(4, 4)
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer split with a different shape
    t = np.arange(20.0).reshape(5, 4)
    input_dict = {"input": t, "indices_or_sections": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: List of indices split
    t = np.arange(16.0).reshape(4, 4)
    input_dict = {"input": t, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Tuple of indices split
    t = np.arange(20.0).reshape(5, 4)
    input_dict = {"input": t, "indices_or_sections": (2, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: 3D Tensor Split
    t = np.arange(24.0).reshape(2, 3, 4)
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Uneven split using list of indices
    t = np.arange(10.0).reshape(10, 1)
    input_dict = {"input": t, "indices_or_sections": [2,5,7]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Splitting a small tensor
    t = np.arange(4.0).reshape(2, 2)
    input_dict = {"input": t, "indices_or_sections": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.vsplit_1"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vsplit_1'.")

check_valid('torch.vsplit', generated_inputs['torch.vsplit_1'], lib="torch")
