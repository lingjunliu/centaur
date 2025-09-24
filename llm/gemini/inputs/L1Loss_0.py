
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def L1Loss_inputs():
    list_of_inputs = []

    input_1 = torch.randn(3, 5, requires_grad=True).detach().numpy()
    target_1 = torch.randn(3, 5).numpy()
    input_dict_1 = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "input": input_1,
        "target": target_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randint(-10, 10, (2, 4), dtype=torch.int32).numpy()
    target_2 = torch.randint(-5, 5, (2, 4), dtype=torch.int32).numpy()
    input_dict_2 = {
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "input": input_2,
        "target": target_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randn(1, 2, 3, 4).numpy()
    target_3 = torch.randn(1, 2, 3, 4).numpy()
    input_dict_3 = {
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input_3,
        "target": target_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randn(size=(4,)).numpy()
    target_4 = torch.randn(size=(4,)).numpy()
    input_dict_4 = {
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "input": input_4,
        "target": target_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_5 = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    target_5 = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()

    input_dict_5 = {
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "input": input_5,
        "target": target_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.L1Loss"] = L1Loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.L1Loss', generated_inputs['torch.nn.L1Loss'], lib="torch")
