
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def std_mean_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dim = (0,)
    unbiased = True
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = (0,)
    unbiased = False
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = (0, 1)
    unbiased = True
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    dim = (0,)
    unbiased = False
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    dim = (0,)
    unbiased = True
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2,2).numpy()
    dim = (0,1)
    unbiased = True
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (1,)
    unbiased = False
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.std_mean_4"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_4'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_4'], lib="torch", suffix=4)
