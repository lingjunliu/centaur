
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def log_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-dimensional input
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type
    input_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Zero value
    input_tensor = torch.tensor([0.00001, 0.0001, 0.001]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger tensor
    input_tensor = torch.rand(2, 3, 4).numpy() * 5
    out_tensor = torch.zeros(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty out tensor with matching shape (important for some implementations)
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out_tensor = torch.empty_like(torch.tensor([1.0, 2.0, 3.0])).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.log"] = log_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log'.")

check_valid('torch.log', generated_inputs['torch.log'], lib="torch", suffix=0)
