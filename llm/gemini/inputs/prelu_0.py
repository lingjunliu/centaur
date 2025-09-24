
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def prelu_inputs():
    list_of_inputs = []

    # Case 1: Scalar weight, 1D input
    input_tensor = torch.randn(5).numpy()
    weight_tensor = torch.tensor(0.25).numpy()
    input_dict = {"input": input_tensor, "weight": weight_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 1D weight, 2D input
    input_tensor = torch.randn(2, 3).numpy()
    weight_tensor = torch.randn(3).numpy()
    input_dict = {"input": input_tensor, "weight": weight_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D weight, 3D input
    input_tensor = torch.randn(2, 3, 4).numpy()
    weight_tensor = torch.randn(3).numpy()
    input_dict = {"input": input_tensor, "weight": weight_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D weight, 4D input, negative input values
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    weight_tensor = torch.randn(3).numpy()
    input_dict = {"input": input_tensor, "weight": weight_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: scalar weight, 4D input
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    weight_tensor = torch.tensor(-0.5).numpy()
    input_dict = {"input": input_tensor, "weight": weight_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.prelu"] = prelu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.prelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.prelu'.")

check_valid('torch.nn.functional.prelu', generated_inputs['torch.nn.functional.prelu'], lib="torch")
