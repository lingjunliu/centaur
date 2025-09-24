
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def CosineSimilarity_inputs():
    list_of_inputs = []

    input1 = torch.randn(100, 128).numpy()
    input2 = torch.randn(100, 128).numpy()
    input_dict = {
        "dim": 1,
        "eps": 1e-6,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(10, 3, 5).numpy()
    input2 = torch.randn(10, 3, 5).numpy()
    input_dict = {
        "dim": 1,
        "eps": 1e-8,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(5, 4, 6, 2).numpy()
    input2 = torch.randn(5, 4, 6, 2).numpy()
    input_dict = {
        "dim": 1,
        "eps": 1e-7,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randn(2, 5, 7).numpy()
    input2 = torch.randn(2, 5, 7).numpy()
    input_dict = {
        "dim": 1,
        "eps": 1e-9,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(1, 5, 3).numpy()
    input2 = torch.randn(1, 5, 3).numpy()
    input_dict = {
        "dim": 1,
        "eps": 1e-8,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randn(2, 3, 4).numpy()
    input2 = torch.randn(2, 3, 1).numpy()
    input_dict = {
        "dim": 1,
        "eps": 1e-8,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.CosineSimilarity"] = CosineSimilarity_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.CosineSimilarity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CosineSimilarity'.")

check_valid('torch.nn.CosineSimilarity', generated_inputs['torch.nn.CosineSimilarity'], lib="torch")
