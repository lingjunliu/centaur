
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_xor_inputs():
    list_of_inputs = []

    # Case 1: Basic integer and tensor
    input_dict = {
        "input": torch.tensor([5], dtype=torch.int32).numpy(),
        "other": torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy(),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Negative integer and tensor
    input_dict = {
        "input": torch.tensor([-3], dtype=torch.int64).numpy(),
        "other": torch.tensor([-1, 0, 1, 2], dtype=torch.int64).numpy(),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Multi-dimensional tensor
    input_dict = {
        "input": torch.tensor([10], dtype=torch.int8).numpy(),
        "other": torch.tensor([[1, 2], [3, 4]], dtype=torch.int8).numpy(),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Different data types (integer and boolean tensor)
    input_dict = {
        "input": torch.tensor([1], dtype=torch.int8).numpy(),
        "other": torch.tensor([True, False, True, False], dtype=torch.bool).numpy().astype(np.int8),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Larger integer
    input_dict = {
        "input": torch.tensor([255], dtype=torch.int16).numpy(),
        "other": torch.tensor([128, 64, 32, 16], dtype=torch.int16).numpy(),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 3D tensor
    input_dict = {
        "input": torch.tensor([7], dtype=torch.int32).numpy(),
        "other": torch.randint(0, 10, (2, 2, 2), dtype=torch.int32).numpy(),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Zero integer and tensor
    input_dict = {
        "input": torch.tensor([0], dtype=torch.int32).numpy(),
        "other": torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy(),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.bitwise_xor_3"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_xor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_xor_3'.")

check_valid('torch.bitwise_xor', generated_inputs['torch.bitwise_xor_3'], lib="torch")
