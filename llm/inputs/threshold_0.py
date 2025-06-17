
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float tensor
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    threshold_val = 0.2
    value_val = 0.0
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int tensor with negative threshold and value
    input_tensor = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.int32)
    threshold_val = -0.5
    value_val = -1.0
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float tensor with different threshold and value
    input_tensor = np.random.randn(2, 3, 4).astype(np.float64)
    threshold_val = 0.7
    value_val = 1.5
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor with threshold equal to value
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    threshold_val = 1.0
    value_val = 1.0
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  4D float tensor with a negative values
    input_tensor = np.random.randn(1, 2, 3, 4).astype(np.float32)
    threshold_val = 0.1
    value_val = -0.5
    input_dict = {"input": input_tensor, "threshold": threshold_val, "value": value_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.threshold'.")

check_valid('torch.threshold', generated_inputs['torch.threshold'], lib="torch")
