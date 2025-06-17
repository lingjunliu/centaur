
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def acosh_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with values >= 1
    input1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with values >= 1
    input2 = np.array([[1.0, 1.2], [2.3, 3.4]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with values >= 1
    input3 = np.array([[[1.0, 1.1], [1.2, 1.3]], [[2.0, 2.1], [2.2, 2.3]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Tensor with one value equal to infinity
    input4 = np.array([1.0, np.inf])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor with various values >= 1
    input5 = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Example with a different data type
    input6 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Example with a scalar value
    input7 = np.array(2.5)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.acosh"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acosh'.")

check_valid('torch.acosh', generated_inputs['torch.acosh'], lib="torch")
