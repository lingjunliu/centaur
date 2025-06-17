
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def logaddexp_inputs():
    list_of_inputs = []

    input1 = np.array([-1.0]).astype(np.float32)
    other1 = np.array([-1.0, -2, -3]).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-100.0, -200, -300]).astype(np.float64)
    other2 = np.array([-1.0, -2, -3]).astype(np.float64)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 2000, 30000]).astype(np.float32)
    other3 = np.array([-1.0, -2, -3]).astype(np.float32)
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1, 2, 3]).astype(np.float32)
    other4 = np.array([4, 5, 6]).astype(np.float32)
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32)
    other5 = np.array([[5.0, 6.0], [7.0, 8.0]]).astype(np.float32)
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logaddexp"] = logaddexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logaddexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logaddexp'.")

check_valid('torch.logaddexp', generated_inputs['torch.logaddexp'], lib="torch")
