
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def isclose_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([1.0 + 1e-7, 2.0, 3.1])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-5,
        "atol": 1e-8,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([float('inf'), 4.0])
    input2 = np.array([float('inf'), 6.0])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 0.5,
        "atol": 1e-8,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([float('nan'), 4.0])
    input2 = np.array([float('nan'), 4.0])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-5,
        "atol": 1e-8,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 2, 4], dtype=np.int32)
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-5,
        "atol": 1,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[1.0, 2.1], [2.9, 4.0]])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 0.1,
        "atol": 1e-8,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.isclose"] = isclose_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isclose'.")

check_valid('torch.isclose', generated_inputs['torch.isclose'], lib="torch")
