
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def local_response_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    size_val = 3
    alpha_val = 0.0001
    beta_val = 0.75
    k_val = 1.0
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    size_val = 5
    alpha_val = 0.0002
    beta_val = 0.5
    k_val = 2.0
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[-1.0, -2.0], [-3.0, -4.0]]]], dtype=np.float32)
    size_val = 3
    alpha_val = 0.0001
    beta_val = 0.75
    k_val = 1.0
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 3, 32, 32).astype(np.float32)
    size_val = 5
    alpha_val = 0.0001
    beta_val = 0.75
    k_val = 1.0
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(2, 5, 16, 16).astype(np.float32)
    size_val = 3
    alpha_val = 0.0002
    beta_val = 0.5
    k_val = 2.0
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_tensor = np.random.rand(3, 1, 8, 8).astype(np.float32)
    size_val = 7
    alpha_val = 0.0003
    beta_val = 0.25
    k_val = 0.5
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = np.array([[[[[1.0]]]]], dtype=np.float32)
    size_val = 1
    alpha_val = 0.0001
    beta_val = 0.75
    k_val = 1.0
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.randn(1, 1, 5, 5).astype(np.float32)
    size_val = 3
    alpha_val = 1e-4
    beta_val = 0.75
    k_val = 1.0
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
    size_val = 3
    alpha_val = 0.0005
    beta_val = 0.6
    k_val = 1.5
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(2, 2, 7, 7).astype(np.float32)
    size_val = 9
    alpha_val = 0.0001
    beta_val = 0.8
    k_val = 0.8
    input_dict = {"input": input_tensor, "size": size_val, "alpha": alpha_val, "beta": beta_val, "k": k_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.local_response_norm"] = local_response_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.local_response_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.local_response_norm'.")

check_valid('torch.nn.functional.local_response_norm', generated_inputs['torch.nn.functional.local_response_norm'], lib="torch", suffix=0)
