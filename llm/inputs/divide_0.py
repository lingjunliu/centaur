
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def divide_inputs():
    list_of_inputs = []

    # Input 1: Basic float division
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([0.5, 2.0, 1.5])
    input_dict1 = {"input": input1, "other": other1, "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer division with floor rounding
    input2 = np.array([5, 6, 7])
    other2 = np.array([2, 3, 2])
    input_dict2 = {"input": input2, "other": other2, "rounding_mode": "floor", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values
    input3 = np.array([-10.0, 5.0, -2.0])
    other3 = np.array([2.0, -1.0, 0.5])
    input_dict3 = {"input": input3, "other": other3, "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multi-dimensional array
    input4 = np.array([[1, 2], [3, 4]])
    other4 = np.array([[0.5, 1], [1.5, 2]])
    input_dict4 = {"input": input4, "other": other4, "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar division
    input5 = np.array([10, 20, 30])
    other5 = 2
    input_dict5 = {"input": input5, "other": np.array(other5), "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Integer division with trunc rounding
    input6 = np.array([-5, 6, -7])
    other6 = np.array([2, -3, 2])
    input_dict6 = {"input": input6, "other": other6, "rounding_mode": "trunc", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Mixed int and float
    input7 = np.array([1, 2, 3], dtype=np.int32)
    other7 = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict7 = {"input": input7, "other": other7, "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.divide"] = divide_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.divide'.")

check_valid('torch.divide', generated_inputs['torch.divide'], lib="torch")
