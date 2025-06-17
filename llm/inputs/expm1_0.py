
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def expm1_inputs():
    list_of_inputs = []

    input1 = np.array([0, 1, -1, 2, -2]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.5, 1.5], [-0.5, -1.5]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3, 4]).reshape((2, 2)).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.expm1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.expm1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.expm1'.")

check_valid('torch.expm1', generated_inputs['torch.expm1'], lib="torch")
