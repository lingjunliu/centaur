
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def histc_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 1.0])
    bins1 = 4
    min1 = 0.0
    max1 = 3.0
    input_dict1 = {"input": input1, "bins": bins1, "min": float(min1), "max": float(max1)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, 0.0, 1.0, 2.0])
    bins2 = 5
    min2 = -2.0
    max2 = 3.0
    input_dict2 = {"input": input2, "bins": bins2, "min": float(min2), "max": float(max2)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    bins3 = 3
    min3 = 1.0
    max3 = 4.0
    input_dict3 = {"input": input3, "bins": bins3, "min": float(min3), "max": float(max3)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    bins4 = 5
    min4 = 0.0
    max4 = 6.0
    input_dict4 = {"input": input4, "bins": bins4, "min": float(min4), "max": float(max4)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    bins5 = 6
    min5 = 0.0
    max5 = 1.0
    input_dict5 = {"input": input5, "bins": bins5, "min": float(min5), "max": float(max5)}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.histc"] = histc_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histc'.")

check_valid('torch.histc', generated_inputs['torch.histc'], lib="torch")
