
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def SmoothL1Loss_inputs():
    list_of_inputs = []

    input_np = np.random.randn(3, 5).astype(np.float32)
    target_np = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "beta": 1.0,
        "input": input_np,
        "target": target_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(2, 4, 6).astype(np.float64)
    target_np = np.random.randn(2, 4, 6).astype(np.float64)
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": 'sum',
        "beta": 0.5,
        "input": input_np,
        "target": target_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(1, 3, 5, 7).astype(np.float16)
    target_np = np.random.randn(1, 3, 5, 7).astype(np.float16)
    input_dict = {
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "beta": 2.0,
        "input": input_np,
        "target": target_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(4, 2).astype(np.float32)
    target_np = np.random.randn(4, 2).astype(np.float32)
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "beta": 0.1,
        "input": input_np,
        "target": target_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(2, 3, 4).astype(np.float32)
    target_np = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "beta": 1.5,
        "input": input_np,
        "target": target_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.SmoothL1Loss"] = SmoothL1Loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.SmoothL1Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SmoothL1Loss'.")

check_valid('torch.nn.SmoothL1Loss', generated_inputs['torch.nn.SmoothL1Loss'], lib="torch")
