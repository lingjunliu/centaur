
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def greater_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3], dtype=np.float32)
    other1 = np.array([2, 2, 2], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[0, 3], [2, 5]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-1, 0, 1], dtype=np.int64)
    other3 = np.array([0, 0, 0], dtype=np.int64)
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    other4 = np.array([1.0, 2.5, 4.0], dtype=np.float64)
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 2, 3], dtype=np.int32)
    other5 = 2
    input_dict5 = {"input": input5, "other": np.array([other5], dtype=np.int32), "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    other6 = np.array([2, 4, 1], dtype=np.float32)
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    other7 = np.array([[[0, 3], [2, 5]], [[4, 7], [6, 9]]], dtype=np.int16)
    input_dict7 = {"input": input7, "other": other7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array(5)
    other8 = np.array(3)
    input_dict8 = {"input": input8, "other": other8, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.greater"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.greater' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.greater'.")

check_valid('torch.greater', generated_inputs['torch.greater'], lib="torch")
