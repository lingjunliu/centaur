
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def less_equal_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[2.0, 1.0], [4.0, 3.0]])
    out1 = np.zeros_like(input1)
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[2, 1], [4, 3]], dtype=np.int32)
    out2 = np.zeros_like(input2, dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Scalar 'other'
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other3 = 3.0
    out3 = np.zeros_like(input3)
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Different shapes (but broadcastable)
    input4 = np.array([[1.0, 2.0, 3.0]])
    other4 = np.array([2.0, 2.0, 2.0])
    out4 = np.zeros_like(input4)
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Negative values
    input5 = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    other5 = np.array([[0.0, 1.0], [-2.0, 5.0]])
    out5 = np.zeros_like(input5)
    input_dict5 = {"input": input5, "other": other5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Multi-dimensional arrays
    input6 = np.random.rand(2, 3, 4)
    other6 = np.random.rand(2, 3, 4)
    out6 = np.zeros_like(input6)
    input_dict6 = {"input": input6, "other": other6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: 'other' is a single number
    input7 = np.array([[1, 2], [3, 4]])
    other7 = 2
    out7 = np.zeros_like(input7)
    input_dict7 = {"input": input7, "other": other7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.less_equal_1"] = less_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.less_equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.less_equal_1'.")

check_valid('torch.less_equal', generated_inputs['torch.less_equal_1'], lib="torch")
