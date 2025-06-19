
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def var_mean_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    dim_list = [0]
    unbiased_bool = True
    keepdim_bool = False
    out_tuple = ()

    input_dict = {
        "input": input_tensor,
        "dim": dim_list,
        "unbiased": unbiased_bool,
        "keepdim": keepdim_bool,
        "out": out_tuple
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 3, 4)
    dim_list = [0, 1]
    unbiased_bool = False
    keepdim_bool = True
    out_tuple = ()

    input_dict = {
        "input": input_tensor,
        "dim": dim_list,
        "unbiased": unbiased_bool,
        "keepdim": keepdim_bool,
        "out": out_tuple
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    dim_list = [1]
    unbiased_bool = True
    keepdim_bool = False
    out_tuple = ()
    input_dict = {
        "input": input_tensor,
        "dim": dim_list,
        "unbiased": unbiased_bool,
        "keepdim": keepdim_bool,
        "out": out_tuple
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(2, 2, 2)
    dim_list = [0, 2]
    unbiased_bool = False
    keepdim_bool = True
    out_tuple = ()

    input_dict = {
        "input": input_tensor,
        "dim": dim_list,
        "unbiased": unbiased_bool,
        "keepdim": keepdim_bool,
        "out": out_tuple
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    dim_list = [0]
    unbiased_bool = True
    keepdim_bool = True
    out_tuple = ()
    input_dict = {
        "input": input_tensor,
        "dim": dim_list,
        "unbiased": unbiased_bool,
        "keepdim": keepdim_bool,
        "out": out_tuple
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.var_mean_1"] = var_mean_inputs()

for input_dict in generated_inputs["torch.var_mean_1"]:
    dim_list = input_dict.get('dim')
    if dim_list is not None:
        input_dict['dim'] = tuple(dim_list)
    else:
        input_dict['dim'] = None

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_1'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_1'], lib="torch", suffix=1)
