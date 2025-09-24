
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MSELoss_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 5, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(3, 5).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 4, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(2, 4).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 1, 10, 10, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(1, 1, 10, 10).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 2, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(4, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3, 4, 5, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(2, 3, 4, 5).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": False,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(size=(10,), requires_grad=True).detach().numpy()
    target_tensor = torch.randn(size=(10,)).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(size=(2,2,2), requires_grad=True).detach().numpy()
    target_tensor = torch.randn(size=(2,2,2)).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MSELoss"] = MSELoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MSELoss', generated_inputs['torch.nn.MSELoss'], lib="torch")
