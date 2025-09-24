
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def narrow_copy_inputs():
    list_of_inputs = []

    # Test case 1: 1D tensor, integer type
    input = torch.arange(10).numpy()
    input_dict = {"input": input, "dim": 0, "start": 2, "length": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D tensor, float type
    input = torch.randn(5, 5).numpy()
    input_dict = {"input": input, "dim": 1, "start": 1, "length": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 3D tensor, negative start
    input = torch.randn(3, 4, 5).numpy()
    input_dict = {"input": input, "dim": 0, "start": -2, "length": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: 4D tensor, complex type
    input = torch.complex(torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2)).numpy()
    input_dict = {"input": input, "dim": 2, "start": 0, "length": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: 2D tensor, dim = 0, large length
    input = torch.randn(10, 5).numpy()
    input_dict = {"input": input, "dim": 0, "start": 2, "length": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: 1D tensor, small length
    input = torch.arange(5).float().numpy()
    input_dict = {"input": input, "dim": 0, "start": 0, "length": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.narrow_copy"] = narrow_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.narrow_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.narrow_copy'.")

check_valid('torch.narrow_copy', generated_inputs['torch.narrow_copy'], lib="torch")
