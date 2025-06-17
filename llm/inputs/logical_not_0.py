
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logical_not_inputs():
    list_of_inputs = []

    input1 = np.array([True, False, True, False])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[True, False], [False, True]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logical_not"] = logical_not_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_not'.")

check_valid('torch.logical_not', generated_inputs['torch.logical_not'], lib="torch")
