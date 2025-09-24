
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def conj_inputs():
    list_of_inputs = []

    # Input 1: Complex tensor
    input1 = np.array([-1 + 1j, -2 + 2j, 3 - 3j])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor
    input2 = np.array([-1.0, -2.0, 3.0])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Int tensor
    input3 = np.array([-1, -2, 3])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D complex tensor
    input4 = np.array([[-1 + 1j, -2 + 2j], [3 - 3j, 4 + 4j]])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D float tensor
    input5 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.conj"] = conj_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.conj'.")

check_valid('torch.conj', generated_inputs['torch.conj'], lib="torch")
