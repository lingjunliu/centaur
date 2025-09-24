
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def vector_to_parameters_inputs():
    list_of_inputs = []

    # Case 1: Basic test with float parameters
    params1 = [torch.randn(2, 3), torch.randn(4)]
    vector1 = torch.cat([p.flatten() for p in params1]).numpy()
    input_dict1 = {"vector": vector1, "parameters": params1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Test with integer parameters
    params2 = [torch.randint(0, 10, (2, 2)), torch.randint(0, 5, (3,))]
    vector2 = torch.cat([p.float().flatten() for p in params2]).numpy()
    input_dict2 = {"vector": vector2, "parameters": params2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Test with negative values
    params3 = [torch.randn(2, 3) * -1, torch.randn(4) * -1]
    vector3 = torch.cat([p.flatten() for p in params3]).numpy()
    input_dict3 = {"vector": vector3, "parameters": params3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Test with different parameter shapes
    params4 = [torch.randn(1, 1, 5, 5), torch.randn(10), torch.randn(1, 1)]
    vector4 = torch.cat([p.flatten() for p in params4]).numpy()
    input_dict4 = {"vector": vector4, "parameters": params4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Test with empty parameters
    params5 = []
    vector5 = torch.tensor([]).numpy()
    input_dict5 = {"vector": vector5, "parameters": params5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Test with single parameter
    params6 = [torch.randn(5, 5)]
    vector6 = torch.cat([p.flatten() for p in params6]).numpy()
    input_dict6 = {"vector": vector6, "parameters": params6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: Test with 3D parameters
    params7 = [torch.randn(2, 3, 4), torch.randn(5)]
    vector7 = torch.cat([p.flatten() for p in params7]).numpy()
    input_dict7 = {"vector": vector7, "parameters": params7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.utils.vector_to_parameters"] = vector_to_parameters_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.vector_to_parameters' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.vector_to_parameters'.")

check_valid('torch.nn.utils.vector_to_parameters', generated_inputs['torch.nn.utils.vector_to_parameters'], lib="torch")
