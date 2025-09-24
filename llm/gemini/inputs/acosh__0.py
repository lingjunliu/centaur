
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def acosh_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3], dtype=np.int64)
    input_dict3 = {"input": input3.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 5, 2, 6], dtype=np.int32).reshape(2, 2)
    input_dict4 = {"input": input4.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1.1, 2.1], [3.1, 4.1]], [[5.1, 6.1], [7.1, 8.1]]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.acosh_"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.acosh_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acosh_'.")

check_valid('torch.acosh_', generated_inputs['torch.acosh_'], lib="torch")
