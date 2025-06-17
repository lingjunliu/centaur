
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def BCELoss_inputs():
    list_of_inputs = []

    input_val = torch.randn(3, 2, requires_grad=True).sigmoid().detach().numpy()
    target_val = torch.rand(3, 2, requires_grad=False).numpy()

    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input_val,
        "target": target_val
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(2, 4, requires_grad=True).sigmoid().detach().numpy()
    target_val = torch.rand(2, 4, requires_grad=False).numpy()
    weight_val = torch.rand(2).numpy()

    input_dict = {
        "weight": weight_val,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input_val,
        "target": target_val
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(1, 5, requires_grad=True).sigmoid().detach().numpy()
    target_val = torch.rand(1, 5, requires_grad=False).numpy()

    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "input": input_val,
        "target": target_val
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(4, 1, 1, requires_grad=True).sigmoid().detach().numpy()
    target_val = torch.rand(4, 1, 1, requires_grad=False).numpy()

    input_dict = {
        "weight": None,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "input": input_val,
        "target": target_val
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(2, 3, 4, 5, requires_grad=True).sigmoid().detach().numpy()
    target_val = torch.rand(2, 3, 4, 5, requires_grad=False).numpy()
    weight_val = torch.rand(2, 3, 4).numpy()

    input_dict = {
        "weight": weight_val,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input": input_val,
        "target": target_val
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.BCELoss"] = BCELoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.BCELoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCELoss'.")

check_valid('torch.nn.BCELoss', generated_inputs['torch.nn.BCELoss'], lib="torch")
