
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lcm_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = torch.tensor([5, 10, 15], dtype=torch.int32).numpy()
    other1 = torch.tensor([3, 4, 5], dtype=torch.int32).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Different shapes, but compatible
    input2 = torch.tensor([5, 10, 15], dtype=torch.int64).numpy()
    other2 = torch.tensor([3], dtype=torch.int64).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Scalar inputs
    input3 = torch.tensor(12, dtype=torch.int16).numpy()
    other3 = torch.tensor(18, dtype=torch.int16).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Zero values
    input4 = torch.tensor([0, 5, 0], dtype=torch.int8).numpy()
    other4 = torch.tensor([3, 0, 7], dtype=torch.int8).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Multi-dimensional tensors
    input5 = torch.tensor([[2, 4], [6, 8]], dtype=torch.int32).numpy()
    other5 = torch.tensor([[3, 5], [7, 9]], dtype=torch.int32).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.lcm"] = lcm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lcm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lcm'.")

check_valid('torch.lcm', generated_inputs['torch.lcm'], lib="torch")
