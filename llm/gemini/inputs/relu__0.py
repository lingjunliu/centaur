
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def relu__inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor with positive and negative values
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float tensor with different values
    input_tensor = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, -3.0]], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float tensor
    input_tensor = np.array([[[1.0, -1.0], [0.0, 2.0]], [[-3.0, 4.0], [-5.0, 6.0]]], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int tensor with mixed values
    input_tensor = np.array([-1, 0, 1, -2, 2], dtype=np.int32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int tensor with mixed values
    input_tensor = np.array([[-1, 0, 1], [-2, 2, -3]], dtype=np.int64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar float
    input_tensor = np.array(-5.0, dtype=np.float64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty numpy array
    input_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.relu_"] = relu__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.relu_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.relu_'.")

check_valid('torch.relu_', generated_inputs['torch.relu_'], lib="torch")
