
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def maximum_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = np.array([1, 2, -1, 5]).astype(np.int32)
    input2 = np.array([3, 0, 4, -2]).astype(np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors with different shapes
    input1 = np.array([[1.5, 2.0], [-1.0, 5.5]]).astype(np.float32)
    input2 = np.array([[3.0, 0.0], [4.0, -2.5]]).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Mixed integer and float tensors
    input1 = np.array([1, 2, -1]).astype(np.int64)
    input2 = np.array([3.5, 0.0, 4.2]).astype(np.float64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensors
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Tensors with negative values and different dtypes
    input1 = np.array([-1.5, -2.0, -3.0]).astype(np.float64)
    input2 = np.array([0, -1, -4]).astype(np.int64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Broadcasting (input1 is scalar)
    input1 = np.array(2.0).astype(np.float32)
    input2 = np.array([1, 3, 0, 2]).astype(np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Broadcasting (input2 is scalar)
    input1 = np.array([1, 3, 0, 2]).astype(np.int32)
    input2 = np.array(2.0).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Boolean Tensors
    input1 = np.array([True, False, True]).astype(np.bool_)
    input2 = np.array([False, True, False]).astype(np.bool_)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.maximum"] = maximum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.maximum'.")

check_valid('torch.maximum', generated_inputs['torch.maximum'], lib="torch")
