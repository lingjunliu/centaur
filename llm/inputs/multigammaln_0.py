
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def multigammaln_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    p1 = 3
    input_dict1 = {
        "input": input1,
        "p": p1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor
    input2 = np.array([6, 7, 8, 9, 10], dtype=np.int32)
    p2 = 2
    input_dict2 = {
        "input": input2,
        "p": p2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Higher dimension tensor
    input3 = np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float64)
    p3 = 1
    input_dict3 = {
        "input": input3,
        "p": p3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Another float tensor with different p
    input4 = np.array([15.5, 16.5, 17.5], dtype=np.float32)
    p4 = 4
    input_dict4 = {
        "input": input4,
        "p": p4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor
    input5 = np.array([[[18.0, 19.0], [20.0, 21.0]], [[22.0, 23.0], [24.0, 25.0]]], dtype=np.float32)
    p5 = 2
    input_dict5 = {
        "input": input5,
        "p": p5,
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
