
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def rsub_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive values
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other_tensor = torch.tensor([4.0, 5.0, 6.0]).numpy()
    alpha_val = 1.0
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Case with negative values and different alpha
    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    other_tensor = torch.tensor([4.0, 5.0, 6.0]).numpy()
    alpha_val = 0.5
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional tensor
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other_tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).numpy()
    alpha_val = 1.0
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero values
    input_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    other_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    alpha_val = 1.0
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Different shapes with broadcasting
    input_tensor = torch.tensor([1.0, 2.0]).numpy()
    other_tensor = torch.tensor([[3.0, 4.0], [5.0, 6.0]]).numpy()
    alpha_val = 1.0
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([4, 5, 6]).numpy()
    alpha_val = 1.0
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.rsub"] = rsub_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rsub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsub'.")

check_valid('torch.rsub', generated_inputs['torch.rsub'], lib="torch", suffix=0)
