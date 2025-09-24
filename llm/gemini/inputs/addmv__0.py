
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def addmv_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    mat_tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=torch.float32)
    vec_tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    beta = 1.0
    alpha = 1.0
    
    input_numpy = input_tensor.numpy()
    mat_numpy = mat_tensor.numpy()
    vec_numpy = vec_tensor.numpy()
    
    input_dict = {"input": input_numpy, "mat": mat_numpy, "vec": vec_numpy, "beta": beta, "alpha": alpha}

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    mat_tensor = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    vec_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32)
    beta = 0.0
    alpha = 2.0
    
    input_numpy = input_tensor.numpy()
    mat_numpy = mat_tensor.numpy()
    vec_numpy = vec_tensor.numpy()
    
    input_dict = {"input": input_numpy, "mat": mat_numpy, "vec": vec_numpy, "beta": beta, "alpha": alpha}
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    mat_tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=torch.float32)
    vec_tensor = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
    beta = 0.5
    alpha = 0.5
    
    input_numpy = input_tensor.numpy()
    mat_numpy = mat_tensor.numpy()
    vec_numpy = vec_tensor.numpy()
    
    input_dict = {"input": input_numpy, "mat": mat_numpy, "vec": vec_numpy, "beta": beta, "alpha": alpha}
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    mat_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32)
    vec_tensor = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    beta = 2.0
    alpha = -1.0
    
    input_numpy = input_tensor.numpy()
    mat_numpy = mat_tensor.numpy()
    vec_numpy = vec_tensor.numpy()
    
    input_dict = {"input": input_numpy, "mat": mat_numpy, "vec": vec_numpy, "beta": beta, "alpha": alpha}
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    mat_tensor = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    vec_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    beta = 1.0
    alpha = 1.0
    
    input_numpy = input_tensor.numpy()
    mat_numpy = mat_tensor.numpy()
    vec_numpy = vec_tensor.numpy()
    
    input_dict = {"input": input_numpy, "mat": mat_numpy, "vec": vec_numpy, "beta": beta, "alpha": alpha}
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.addmv_"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addmv_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv_'.")

check_valid('torch.addmv_', generated_inputs['torch.addmv_'], lib="torch", suffix=0)
