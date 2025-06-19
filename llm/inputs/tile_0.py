
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def tile_inputs():
    list_of_inputs = []

    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": x,
        "dims": (2,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {
        "input": y,
        "dims": (2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    z = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": z,
        "dims": (1, 2, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = torch.randint(0, 10, (5,)).numpy()
    input_dict = {
        "input": a,
        "dims": (3,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    b = torch.randn(1, 5, 5, 5).numpy()
    input_dict = {
        "input": b,
        "dims": (2, 1, 1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    c = torch.tensor([[-1, 2], [-3, 4]]).numpy()
    input_dict = {
        "input": c,
        "dims": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    d = torch.randn(3, 2).numpy()
    input_dict = {
        "input": d,
        "dims": (3, 3, 2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    e = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": e,
        "dims": (2,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    f = torch.randn(size=(2,3,4)).numpy()
    input_dict = {
        "input": f,
        "dims": (2,2,2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tile"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tile'.")

check_valid('torch.tile', generated_inputs['torch.tile'], lib="torch")
