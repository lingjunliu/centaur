
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def var_mean_inputs():
    list_of_inputs = []

    # Input 1
    input_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = True
    dim = (0,)
    keepdim = False
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = False
    dim = (0,)
    keepdim = True
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_np = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = True
    dim = (1,)
    keepdim = False
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2)
    input = torch.from_numpy(input_np)
    unbiased = False
    dim = (0, 1)
    keepdim = True
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_np = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = True
    dim = (0,)
    keepdim = True
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_np = np.array([1, 2, 3], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = False
    dim = (0,)
    keepdim = False
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = True
    dim = (0, 1)
    keepdim = False
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = False
    dim = (0, 2)
    keepdim = True
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input = torch.from_numpy(input_np)
    unbiased = True
    dim = ()
    keepdim = False
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(2,3)
    input = torch.from_numpy(input_np)
    unbiased = False
    dim = (0,)
    keepdim = False
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(2,3)
    input = torch.from_numpy(input_np)
    unbiased = True
    dim = (1,)
    keepdim = True
    out = None
    input_dict = {"input": input, "dim": dim, "unbiased": unbiased, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.var_mean_4"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_4'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_4'], lib="torch", suffix=4)
