
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_sum_inputs():
    list_of_inputs = []

    # Test case 1: Sum of all elements
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Sum along a specific dimension (dim=1)
    input_tensor = torch.randn(3, 5).numpy()
    input_dict = {"input": input_tensor, "dim": (1,), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Sum along multiple dimensions (dim=(0, 2))
    input_tensor = torch.randn(4, 2, 3).numpy()
    input_dict = {"input": input_tensor, "dim": (0, 2), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Sum with keepdim=True
    input_tensor = torch.randn(2, 2, 2).numpy()
    input_dict = {"input": input_tensor, "dim": (1,), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Sum with dtype specified (torch.float64)
    input_tensor = torch.randint(0, 10, (3, 3)).numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Sum of a 1D tensor
    input_tensor = torch.arange(5).numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Sum with negative values
    input_tensor = torch.randint(-5, 5, (2, 4)).float().numpy()
    input_dict = {"input": input_tensor, "dim": (0,), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 8: Sum with all dimensions
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim": (0, 1, 2), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sum_3"] = torch_sum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sum_3'.")

check_valid('torch.sum', generated_inputs['torch.sum_3'], lib="torch")
