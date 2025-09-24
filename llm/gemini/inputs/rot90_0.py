
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def rot90_inputs():
    list_of_inputs = []

    # Input 1: 2D integer tensor, default k and dims
    x = np.arange(4).reshape(2, 2)
    input_dict = {
        "input": x,
        "k": 1,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float tensor, k=2, dims=(0, 2)
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {
        "input": x,
        "k": 2,
        "dims": (0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D complex tensor, k=-1, dims=(1, 3)
    x = (np.arange(120) + 1j * np.arange(120)).reshape(2, 3, 4, 5)
    input_dict = {
        "input": x,
        "k": -1,
        "dims": (1, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor, k=0, dims=(0, 1) - no rotation
    x = np.array([[1, 2], [3, 4]])
    input_dict = {
        "input": x,
        "k": 0,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D integer tensor, k=3, dims=(0, 1)
    x = np.arange(8).reshape(2, 2, 2)
    input_dict = {
        "input": x,
        "k": 3,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.rot90"] = rot90_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rot90'.")

check_valid('torch.rot90', generated_inputs['torch.rot90'], lib="torch")
