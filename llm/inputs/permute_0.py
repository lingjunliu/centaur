
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def permute_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D tensor with a simple permutation
    x = torch.randn(2, 3, 5).numpy()
    dims = (2, 0, 1)
    input_dict = {"input": x, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D tensor with a more complex permutation
    x = torch.randn(2, 3, 4, 5).numpy()
    dims = (3, 0, 2, 1)
    input_dict = {"input": x, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor (matrix) with transposed dimensions
    x = torch.randn(5, 7).numpy()
    dims = (1, 0)
    input_dict = {"input": x, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor (vector) - permuting it should not change anything
    x = torch.randn(10).numpy()
    dims = (0,)
    input_dict = {"input": x, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Int Tensor
    x = torch.randint(0, 10, (2, 3, 4)).numpy()
    dims = (1, 2, 0)
    input_dict = {"input": x, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty Tensor
    x = torch.empty(0).numpy()
    dims = (0,)  # Technically, permuting an empty tensor is valid, but it often leads to errors down stream
    input_dict = {"input": x, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 Tensor
    x = torch.randn(2, 3, 5, dtype=torch.float64).numpy()
    dims = (2, 0, 1)
    input_dict = {"input": x, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.permute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.permute'.")

check_valid('torch.permute', generated_inputs['torch.permute'], lib="torch")
