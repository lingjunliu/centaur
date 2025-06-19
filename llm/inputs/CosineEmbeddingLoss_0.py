
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def CosineEmbeddingLoss_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 5, requires_grad=True).detach().numpy()
    input2 = torch.randn(3, 5, requires_grad=True).detach().numpy()
    target = torch.ones(3).numpy()
    input_dict = {
        "margin": 0.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 4, requires_grad=True).detach().numpy()
    input2 = torch.randn(2, 4, requires_grad=True).detach().numpy()
    target = torch.tensor([-1, 1]).numpy()
    input_dict = {
        "margin": 0.5,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(1, 3, requires_grad=True).detach().numpy()
    input2 = torch.randn(1, 3, requires_grad=True).detach().numpy()
    target = torch.tensor([-1]).numpy()
    input_dict = {
        "margin": 0.2,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(4, 2, requires_grad=True).detach().numpy()
    input2 = torch.randn(4, 2, requires_grad=True).detach().numpy()
    target = torch.tensor([1, -1, 1, -1]).numpy()
    input_dict = {
        "margin": 0.8,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 5, requires_grad=True).detach().numpy()
    input2 = torch.randn(2, 5, requires_grad=True).detach().numpy()
    target = torch.tensor([1, 1]).numpy()
    input_dict = {
        "margin": 0.3,
        "size_average": True,
        "reduce": False,
        "reduction": 'sum',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.CosineEmbeddingLoss"] = CosineEmbeddingLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.CosineEmbeddingLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CosineEmbeddingLoss'.")

check_valid('torch.nn.CosineEmbeddingLoss', generated_inputs['torch.nn.CosineEmbeddingLoss'], lib="torch")
