
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logit_inputs():
    list_of_inputs = []

    input_dict = {
        "input": np.array([0.1, 0.5, 0.9]).astype(np.float32),
        "eps": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[0.01, 0.5], [0.7, 0.99]]).astype(np.float64),
        "eps": 1e-6,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[[0.2, 0.4], [0.6, 0.8]], [[0.1, 0.3], [0.7, 0.9]]]).astype(np.float32),
        "eps": 1e-4,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([0.001, 0.999]).astype(np.float64),
        "eps": 1e-9,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.2, 0.3, 0.4, 0.5]).astype(np.float32),
        "eps": 0.01,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.array([[0.0001, 0.9999], [0.5, 0.5]]).astype(np.float64),
        "eps": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.logit"] = logit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logit'.")

check_valid('torch.logit', generated_inputs['torch.logit'], lib="torch")
