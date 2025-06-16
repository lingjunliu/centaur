
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def MultiMarginLoss_inputs():
    list_of_inputs = []

    input1 = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target1 = torch.tensor([3]).numpy()
    input_dict1 = {
        "p": 1,
        "margin": 1.0,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 5).numpy()
    target2 = torch.randint(0, 5, (2,)).numpy()
    weight2 = torch.randn(5).numpy()
    input_dict2 = {
        "p": 2,
        "margin": 0.5,
        "weight": weight2,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(10, 3).numpy()
    target3 = torch.randint(0, 3, (10,)).numpy()
    input_dict3 = {
        "p": 1,
        "margin": 2.0,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 4).numpy()
    target4 = torch.tensor([0]).numpy()
    weight4 = torch.tensor([0.2, 0.3, 0.1, 0.4]).numpy()
    input_dict4 = {
        "p": 1,
        "margin": 0.75,
        "weight": weight4,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5, 2).numpy()
    target5 = torch.tensor([0, 1, 0, 1, 0]).numpy()
    input_dict5 = {
        "p": 2,
        "margin": 1.5,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MultiMarginLoss"] = MultiMarginLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MultiMarginLoss', generated_inputs['torch.nn.MultiMarginLoss'], lib="torch")
