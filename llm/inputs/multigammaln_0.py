
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def multigammaln_inputs():
    list_of_inputs = []

    input1 = np.random.rand(3, 4).astype(np.float32) + 1
    p1 = 2
    input_dict1 = {
        "input": input1,
        "p": p1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(5).astype(np.float64) + 1
    p2 = 3
    input_dict2 = {
        "input": input2,
        "p": p2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = (np.random.rand(2, 2, 2) * 5).astype(np.float32) + 1
    p3 = 1
    input_dict3 = {
        "input": input3,
        "p": p3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = (np.random.rand(1, 1, 1, 1) * 10).astype(np.float64) + 1
    p4 = 4
    input_dict4 = {
        "input": input4,
        "p": p4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = (np.random.rand(4) * 2).astype(np.float32) + 1
    p5 = 1
    input_dict5 = {
        "input": input5,
        "p": p5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.special.multigammaln"] = multigammaln_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.multigammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.multigammaln'.")

check_valid('torch.special.multigammaln', generated_inputs['torch.special.multigammaln'], lib="torch")
