
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def arccosh_inputs():
    list_of_inputs = []

    # Input 1: Scalar value greater than or equal to 1
    input_1 = np.array(2.0)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D array with values >= 1
    input_2 = np.array([1.0, 2.0, 3.0])
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D array with values >= 1
    input_3 = np.array([[1.0, 1.5], [2.0, 2.5]])
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: float32 array
    input_4 = np.array([1.0, 2.0], dtype=np.float32)
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: float64 array
    input_5 = np.array([3.0], dtype=np.float64)
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.arccosh"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arccosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arccosh'.")

check_valid('torch.arccosh', generated_inputs['torch.arccosh'], lib="torch")
