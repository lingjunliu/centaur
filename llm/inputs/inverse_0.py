
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def inverse_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix (float)
    input1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Square matrix with negative values (float)
    input2 = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Another square matrix
    input3 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Square matrix with small values
    input4 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Diagonal matrix
    input5 = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.inverse"] = inverse_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.inverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.inverse'.")

check_valid('torch.inverse', generated_inputs['torch.inverse'], lib="torch")
