
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_round_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_tensor = torch.tensor([4.7, -2.3, 9.1, -7.7]).float().numpy()
    decimals = 0
    input_dict = {"input": input_tensor, "decimals": decimals, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor with equidistant values, decimals=0
    input_tensor = torch.tensor([-0.5, 0.5, 1.5, 2.5]).float().numpy()
    decimals = 0
    input_dict = {"input": input_tensor, "decimals": decimals, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with decimals > 0
    input_tensor = torch.tensor([0.1234567]).float().numpy()
    decimals = 3
    input_dict = {"input": input_tensor, "decimals": decimals, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with decimals < 0
    input_tensor = torch.tensor([1200.1234567]).float().numpy()
    decimals = -3
    input_dict = {"input": input_tensor, "decimals": decimals, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer tensor as float
    input_tensor = torch.tensor([1, 2, 3, 4, 5]).float().numpy()
    decimals = 0
    input_dict = {"input": input_tensor, "decimals": decimals, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.round"] = torch_round_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.round'.")

check_valid('torch.round', generated_inputs['torch.round'], lib="torch")
