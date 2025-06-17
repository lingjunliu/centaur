
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def relu6_inputs():
    list_of_inputs = []

    input = np.random.randn(3, 4).astype(np.float32)
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(2, 3, 4).astype(np.float64)
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([-1, 0, 1, 2, 7, -2]).astype(np.int32)
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[-1.5, 0.5, 1.5], [2.5, 7.5, -2.5]]).astype(np.float32)
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 5, 5, 5).astype(np.float32)
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array(5).astype(np.int64)
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.relu6"] = relu6_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.relu6'.")

check_valid('torch.nn.functional.relu6', generated_inputs['torch.nn.functional.relu6'], lib="torch")
