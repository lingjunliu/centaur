
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def flip_inputs():
    list_of_inputs = []

    # Test case 1: 2D float tensor
    input = torch.randn(2, 3).numpy()
    dims = [0]
    input_dict = {"input": input, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 3D int tensor
    input = torch.randint(0, 10, (2, 2, 2)).numpy()
    dims = [0, 1]
    input_dict = {"input": input, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 4D complex tensor
    input = torch.randn(2, 3, 4, 2, dtype=torch.complex64).numpy()
    dims = [1, 3]
    input_dict = {"input": input, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: 1D tensor with a single dimension to flip
    input = torch.arange(5).numpy()
    dims = [0]
    input_dict = {"input": input, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: 5D tensor with multiple dimensions
    input = torch.randn(2, 3, 2, 4, 2).numpy()
    dims = [0, 2, 4]
    input_dict = {"input": input, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Negative dimension index
    input = torch.randn(2, 3, 4).numpy()
    dims = [-1]
    input_dict = {"input": input, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Empty list of dimensions
    input = torch.randn(2, 3).numpy()
    dims = []
    input_dict = {"input": input, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.flip"] = flip_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.flip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flip'.")

check_valid('torch.flip', generated_inputs['torch.flip'], lib="torch")
