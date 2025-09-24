
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def arctanh_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D, values within (-1, 1)
    input1 = np.array([-0.5, 0.2, 0.8, -0.9], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, 2D, values within (-1, 1)
    input2 = np.array([[0.1, -0.3], [0.6, -0.7]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor, 3D, values within (-1, 1)
    input3 = np.array([[[0.2, -0.4], [0.5, -0.8]], [[0.9, -0.1], [-0.6, 0.3]]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor, scalar, values within (-1, 1)
    input4 = np.array(0.4, dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Float tensor, 1D, including 0
    input5 = np.array([-0.2, 0, 0.5, -0.8], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float tensor, 2D, some values close to 1 and -1
    input6 = np.array([[0.99, -0.9], [0.5, -0.999]], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.arctanh_"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arctanh_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arctanh_'.")

check_valid('torch.arctanh_', generated_inputs['torch.arctanh_'], lib="torch")
