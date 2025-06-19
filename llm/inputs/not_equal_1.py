
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def not_equal_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 4, 3], dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors with different values
    input1 = np.array([1.0, 2.5, 3.2], dtype=np.float32)
    input2 = np.array([1.5, 2.5, 3.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Multi-dimensional tensors
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input2 = np.array([[1, 5], [6, 4]], dtype=np.int64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Mixed types (int and float)
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1.0, 2.0, 3.5], dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Negative values
    input1 = np.array([-1, -2, 3], dtype=np.int32)
    input2 = np.array([1, -2, -3], dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Complex numbers
    input1 = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    input2 = np.array([1 + 1j, 2 + 3j, 3 + 4j], dtype=np.complex64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Zero dimensional tensors
    input1 = np.array(1, dtype=np.int32)
    input2 = np.array(2, dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.not_equal_1"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.not_equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.not_equal_1'.")

check_valid('torch.not_equal', generated_inputs['torch.not_equal_1'], lib="torch")
