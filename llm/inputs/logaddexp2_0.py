
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logaddexp2_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4).astype(np.float32)
    input2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(2, 2, 2).astype(np.float64)
    input2 = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict = {
        "input": input1,
        "other": input2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randint(-5, 5, size=(5,)).astype(np.float32)
    input2 = np.random.randint(-5, 5, size=(5,)).astype(np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array(1.0).astype(np.float32)
    input2 = np.array(2.0).astype(np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(10).astype(np.float32)
    input2 = np.random.randn(10).astype(np.float32)
    out = np.zeros(10).astype(np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logaddexp2"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logaddexp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logaddexp2'.")

check_valid('torch.logaddexp2', generated_inputs['torch.logaddexp2'], lib="torch")
