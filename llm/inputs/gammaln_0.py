
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gammaln_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    input1 = np.array(2.5, dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D array of floats
    input2 = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D array of floats
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D array of floats
    input4 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Array with small positive numbers close to zero (to test behavior near zero).
    input5 = np.array([0.1, 0.01, 0.001, 0.0001], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.special.gammaln"] = gammaln_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.gammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.gammaln'.")

check_valid('torch.special.gammaln', generated_inputs['torch.special.gammaln'], lib="torch")
