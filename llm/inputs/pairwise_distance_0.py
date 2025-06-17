
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def pairwise_distance_inputs():
    list_of_inputs = []

    x1 = torch.randn(10, 5).numpy()
    x2 = torch.randn(10, 5).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False

    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(5, 3, 2).numpy()
    x2 = torch.randn(5, 3, 2).numpy()
    p = 1.5
    eps = 1e-8
    keepdim = True

    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randint(-5, 5, (3, 4)).float().numpy()
    x2 = torch.randint(-5, 5, (3, 4)).float().numpy()
    p = 3.0
    eps = 1e-4
    keepdim = False

    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(2, 2, 2, 2).numpy()
    x2 = torch.randn(2, 2, 2, 2).numpy()
    p = 0.5
    eps = 1e-12
    keepdim = True

    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = torch.randn(7, 1).numpy()
    x2 = torch.randn(7, 1).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False

    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.pairwise_distance"] = pairwise_distance_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.pairwise_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pairwise_distance'.")

check_valid('torch.nn.functional.pairwise_distance', generated_inputs['torch.nn.functional.pairwise_distance'], lib="torch")
