
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_t_inputs():
    list_of_inputs = []

    # 1-D tensor (vector)
    x = torch.randn(5).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2-D tensor (matrix)
    x = torch.randn(2, 3).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2-D tensor with integer type
    x = torch.randint(-10, 10, size=(4, 2)).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2-D tensor with negative values
    x = (torch.randn(3, 4) * -1).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2-D tensor with all zeros
    x = torch.zeros((2, 2)).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.t"] = torch_t_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.t' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.t'.")

check_valid('torch.t', generated_inputs['torch.t'], lib="torch")
