
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic case with dim=0
    x = torch.randn(2, 3).numpy()
    tensors = [x, x]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dim value
    x = torch.randn(2, 3).numpy()
    tensors = [x, x]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shape tensors
    x = torch.randn(1, 5).numpy()
    y = torch.randn(1, 5).numpy()
    tensors = [x, y]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Stacking more than two tensors
    x = torch.randn(3, 4).numpy()
    y = torch.randn(3, 4).numpy()
    z = torch.randn(3, 4).numpy()
    tensors = [x, y, z]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative dim
    x = torch.randn(2, 3).numpy()
    tensors = [x, x]
    dim = -1
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors
    x = torch.randn(2, 3, 4).numpy()
    tensors = [x, x]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int tensors
    x = torch.randint(0, 10, (2, 3)).numpy()
    tensors = [x, x]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.stack"] = torch_stack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.stack'.")

check_valid('torch.stack', generated_inputs['torch.stack'], lib="torch")
