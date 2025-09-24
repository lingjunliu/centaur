
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def arccos_inputs():
    list_of_inputs = []

    input1 = np.array([0.5, -0.2, 0.9]).astype(np.float32)
    input_dict1 = {"input": input1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float64)
    input_dict2 = {"input": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0]]).astype(np.float32)
    input_dict3 = {"input": input3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, 0.0, -1.0]).astype(np.float64)
    input_dict4 = {"input": input4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[0.2, 0.4], [0.6, 0.8]], [[-0.2, -0.4], [-0.6, -0.8]]]).astype(np.float32)
    input_dict5 = {"input": input5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([0]).astype(np.float32)
    input_dict6 = {"input": input6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([-1]).astype(np.float64)
    input_dict7 = {"input": input7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.arccos"] = arccos_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arccos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arccos'.")

check_valid('torch.arccos', generated_inputs['torch.arccos'], lib="torch")
