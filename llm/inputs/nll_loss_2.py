
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def nll_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float input and long target
    input = np.array([[-0.8, -0.2, -0.9], [-0.3, -0.1, -0.5]], dtype=np.float32)
    target = np.array([0, 1], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "log_target": None,
        "weight": None,
        "size_average": None,
        "ignore_index": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: With weight and ignore_index
    input = np.array([[-0.8, -0.2, -0.9], [-0.3, -0.1, -0.5]], dtype=np.float32)
    target = np.array([0, 1], dtype=np.int64)
    weight = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    input_dict = {
        "input": input,
        "target": target,
        "log_target": None,
        "weight": weight,
        "size_average": None,
        "ignore_index": 1,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.nll_loss_2"] = nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.nll_loss_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.nll_loss_2'.")

check_valid('torch.nn.functional.nll_loss', generated_inputs['torch.nn.functional.nll_loss_2'], lib="torch")
