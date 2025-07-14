
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def zeta_inputs():
    list_of_inputs = []

    # Input 1: Basic positive tensor
    x = torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor with integers
    x = torch.tensor([2, 3, 4], dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with larger values
    x = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional tensor
    x = torch.tensor([[1.5, 2.0], [2.5, 3.0]], dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with ones
    x = (torch.ones((2, 2), dtype=torch.float32) + 1.1).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with small positive values
    x = torch.tensor([1.01, 1.05, 1.1], dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with different sized dimensions
    x = (torch.rand(2, 3, 4, dtype=torch.float32).numpy() * 2) + 2  # Ensure all values > 1
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with very large values
    x = torch.tensor([1000.0, 2000.0, 3000.0], dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with a scalar
    x = torch.tensor(5.0, dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Tensor with floating point numbers close to 1
    x = (torch.rand(5, dtype=torch.float32).numpy() * 0.1) + 1.0001
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Tensor with shape (0,)
    x = torch.empty((0,), dtype=torch.float32).numpy()
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.zeta_3"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.zeta_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.zeta_3'.")

check_valid('torch.special.zeta', generated_inputs['torch.special.zeta_3'], lib="torch", suffix=3)
