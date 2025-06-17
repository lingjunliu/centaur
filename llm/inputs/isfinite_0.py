
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def isfinite_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor with inf, -inf, nan, and finite values
    input1 = np.array([1.0, float('inf'), 2.0, float('-inf'), float('nan'), 3.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with a mix of values
    input2 = np.array([[1.0, 2.0, float('inf')], [float('-inf'), float('nan'), 4.0]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D int tensor
    input3 = np.array([1, 2, 3, -4, 0], dtype=np.int32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 3D float tensor
    input4 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Complex tensor with finite and infinite values in real and imaginary parts
    input5 = np.array([1 + 1j, float('inf') + 2j, 3 - float('inf') * 1j, float('nan') + 4j], dtype=np.complex64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

def convert_to_torch(generated_inputs):
    for api_name, input_list in generated_inputs.items():
        for input_dict in input_list:
            for key, value in input_dict.items():
                if isinstance(value, np.ndarray):
                    input_dict[key] = torch.from_numpy(value)
                elif isinstance(value, (int, float)):
                    input_dict[key] = torch.tensor(value)
    return generated_inputs

generated_inputs["torch.isfinite"] = isfinite_inputs()
generated_inputs = convert_to_torch(generated_inputs)

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isfinite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isfinite'.")

check_valid('torch.isfinite', generated_inputs['torch.isfinite'], lib="torch")
