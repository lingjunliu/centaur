
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []

    # Case 1: Integer tensors
    input1 = np.array([5, 6, 7, 8])
    other1 = np.array([2, 3, 2, 1])
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Float tensors
    input2 = np.array([5.5, 6.6, 7.7, 8.8])
    other2 = np.array([2.0, 3.0, 2.0, 1.0])
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Negative values
    input3 = np.array([-5, 6, -7, 8])
    other3 = np.array([2, -3, 2, -1])
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Broadcasting (scalar)
    input4 = np.array([5, 6, 7, 8])
    other4 = np.array(2)
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Multi-dimensional array
    input5 = np.array([[5, 6], [7, 8]])
    other5 = np.array([[2, 3], [2, 1]])
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Multi-dimensional array with float
    input6 = np.array([[5.0, 6.0], [7.0, 8.0]])
    other6 = np.array([[2.0, 3.0], [2.0, 1.0]])
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: Broadcasting with different shapes
    input7 = np.array([[10, 20, 30], [40, 50, 60]])
    other7 = np.array([2, 5, 10])
    input_dict7 = {"input": input7, "other": other7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.floor_divide_1"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.floor_divide_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_1'.")

check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_1'], lib="torch")
