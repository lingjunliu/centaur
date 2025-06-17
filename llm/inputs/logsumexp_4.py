
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor, dim=1, keepdim=False
    input1 = torch.randn(3, 3).numpy()
    dim1 = (1,)
    keepdim1 = False
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor, dim=0, keepdim=True
    input2 = torch.randint(-5, 5, (2, 4)).float().numpy()
    dim2 = (0,)
    keepdim2 = True
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.logsumexp_4"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logsumexp_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_4'.")

check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_4'], lib="torch")
