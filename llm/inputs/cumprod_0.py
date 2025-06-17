
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cumprod_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor, dim=0, float
    input_tensor = torch.randn(5).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 2D tensor, dim=1, int
    input_tensor = torch.randint(-5, 5, (3, 4)).numpy()
    dim = 1
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D tensor, dim=2, float with dtype specified
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 2
    dtype = torch.float64
    input_dict = {"input": input_tensor, "dim": dim}
    if 'dtype' in locals():
        input_dict['dtype'] = dtype
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 1D tensor with negative values, dim=0, int
    input_tensor = torch.randint(-10, -1, (6,)).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: 2D tensor, dim=0, float with out specified
    input_tensor = torch.randn(4, 5).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cumprod"] = cumprod_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cumprod'.")

check_valid('torch.cumprod', generated_inputs['torch.cumprod'], lib="torch")
