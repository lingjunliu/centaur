
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensor and float
    input_tensor = np.array([5, 2, 8, -3, 10]).astype(np.int32)
    other_value = 2.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensor and float
    input_tensor = np.array([5.5, 2.2, 8.8, -3.3, 10.1]).astype(np.float32)
    other_value = 2.5
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D integer tensor and float
    input_tensor = np.array([[5, 2], [8, -3], [10, 4]]).astype(np.int64)
    other_value = 3.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D float tensor and float
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64) * 10
    other_value = 1.5
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values in tensor and float
    input_tensor = np.array([-5, -2, 8, -3, 10]).astype(np.int8)
    other_value = -2.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Large values
    input_tensor = np.array([1000, 2000, 3000]).astype(np.int32)
    other_value = 500.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Zero value in tensor
    input_tensor = np.array([5, 0, 8, -3, 10]).astype(np.int16)
    other_value = 2.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.floor_divide_2"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.floor_divide_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_2'.")

check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_2'], lib="torch")
