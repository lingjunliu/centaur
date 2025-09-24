
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def broadcast_shapes_inputs():
    list_of_inputs = []

    input_dict = {
        "shape1": (5, 4),
        "shape2": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "shape1": (5, 4),
        "shape2": (4,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "shape1": (15, 3, 5),
        "shape2": (15, 1, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "shape1": (15, 3, 5),
        "shape2": (3, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "shape1": (15, 3, 5),
        "shape2": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "shape1": (10, 1, 5),
        "shape2": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape1": (2, 3, 4, 5),
        "shape2": (3, 4, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape1": (2, 3, 1, 5),
        "shape2": (3, 4, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.broadcast_shapes"] = broadcast_shapes_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.broadcast_shapes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.broadcast_shapes'.")

check_valid('torch.broadcast_shapes', generated_inputs['torch.broadcast_shapes'], lib="torch")
