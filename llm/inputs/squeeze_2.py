
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_squeeze_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 1, 3, 1, 4).numpy()
    dim = 1
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5, 1, 2).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(3, 1, 2, 1).numpy()
    dim = 3
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 4, 1, 6).numpy()
    dim = -1
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 2, 1, 3, 1).numpy()
    dim = 2
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 1, 1, 1).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randint(0, 10, (2, 1, 3, 1, 4)).numpy()
    dim = 1
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.squeeze_2"] = torch_squeeze_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.squeeze_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.squeeze_2'.")

check_valid('torch.squeeze', generated_inputs['torch.squeeze_2'], lib="torch")
