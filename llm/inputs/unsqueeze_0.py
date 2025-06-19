
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unsqueeze_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, dim = 0
    x = torch.tensor([1, 2, 3, 4]).numpy()
    dim = 0
    input_dict = {"input": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, dim = 1
    x = torch.randn(2, 3).numpy()
    dim = 1
    input_dict = {"input": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, dim = -1
    x = torch.randint(0, 10, (2, 3, 4)).numpy()
    dim = -1
    input_dict = {"input": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D tensor, dim = -4 (minimum dim)
    x = torch.randn(1, 2, 3, 4).numpy()
    dim = -4
    input_dict = {"input": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor, dim = input.dim()
    x = torch.randn(3, 5).numpy()
    dim = 2
    input_dict = {"input": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float tensor, dim = 0
    x = torch.randn(5).numpy()
    dim = 0
    input_dict = {"input": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Int tensor, dim = 1
    x = torch.randint(0, 10, (5,)).numpy()
    dim = 1
    input_dict = {"input": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.unsqueeze"] = unsqueeze_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unsqueeze' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unsqueeze'.")

check_valid('torch.unsqueeze', generated_inputs['torch.unsqueeze'], lib="torch")
