
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def arcsin__inputs():
    list_of_inputs = []

    input1 = np.array([0.5, -0.2, 0.9, -0.8, 0.0]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.1, 0.2], [-0.3, 0.4]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[0.5], [-0.5]], [[0.8], [-0.2]]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, -1.0, 0.0]).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-0.5, 0.5, -0.9, 0.9]).reshape(2, 2).astype(np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.arcsin_"] = arcsin__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arcsin_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arcsin_'.")

check_valid('torch.arcsin_', generated_inputs['torch.arcsin_'], lib="torch")
