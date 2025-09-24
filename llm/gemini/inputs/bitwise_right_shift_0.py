
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_right_shift_inputs():
    list_of_inputs = []

    # Test case 1: Basic integer tensor
    input1 = np.array([10, 20, 30], dtype=np.int32)
    other1 = np.array([1, 2, 3], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Different integer type
    input2 = np.array([255, 128, 64], dtype=np.int8)
    other2 = np.array([2, 1, 0], dtype=np.int8)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Multi-dimensional tensor
    input3 = np.array([[16, 32], [64, 128]], dtype=np.int64)
    other3 = np.array([[0, 1], [2, 3]], dtype=np.int64)
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Negative values
    input4 = np.array([-16, -32, -64], dtype=np.int32)
    other4 = np.array([1, 2, 3], dtype=np.int32)
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Broadcasting
    input5 = np.array([256, 512], dtype=np.int16)
    other5 = np.array(2, dtype=np.int16)
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Zero shifts
    input6 = np.array([1, 2, 3], dtype=np.int32)
    other6 = np.array([0, 0, 0], dtype=np.int32)
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Larger shifts
    input7 = np.array([1024, 2048], dtype=np.int32)
    other7 = np.array([5, 10], dtype=np.int32)
    input_dict7 = {"input": input7, "other": other7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.bitwise_right_shift"] = bitwise_right_shift_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_right_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_right_shift'.")

check_valid('torch.bitwise_right_shift', generated_inputs['torch.bitwise_right_shift'], lib="torch")
