
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def i0e_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    input_dict = {"input": np.array(1.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with positive and negative floats
    input_dict = {"input": np.array([-2.0, -1.0, 0.0, 1.0, 2.0])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with mixed values
    input_dict = {"input": np.array([[0.5, -0.5], [1.0, -1.0]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with larger values
    input_dict = {"input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with zero
    input_dict = {"input": np.array([0.0])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.i0e"] = i0e_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i0e'.")

check_valid('torch.special.i0e', generated_inputs['torch.special.i0e'], lib="torch")
