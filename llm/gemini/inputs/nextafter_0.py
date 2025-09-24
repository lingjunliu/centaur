
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0], dtype=np.float32)
    other1 = np.array([2.0, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -2.0], dtype=np.float64)
    other2 = np.array([-2.0, -1.0], dtype=np.float64)
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other3 = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input5 = np.array([1.0], dtype=np.float32)
    other5 = np.array([np.inf], dtype=np.float32)
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([float('nan')], dtype=np.float32)
    other6 = np.array([1.0], dtype=np.float32)
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other7 = np.array([2.0], dtype=np.float32)
    input_dict7 = {"input": input7, "other": other7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nextafter"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nextafter'.")

check_valid('torch.nextafter', generated_inputs['torch.nextafter'], lib="torch")
