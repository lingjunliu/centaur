
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def trapz_inputs():
    list_of_inputs = []

    y = np.array([1, 2, 3], dtype=np.float32)
    x = np.array([4, 6, 8], dtype=np.float32)
    dim = 0
    input_dict = {"y": y, "x": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32)
    dim = 1
    input_dict = {"y": y, "x": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = np.array([2, 4, 6], dtype=np.float32)
    dim = 1
    input_dict = {"y": y, "x": x, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.trapz"] = trapz_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.trapz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.trapz'.")

check_valid('torch.trapz', generated_inputs['torch.trapz'], lib="torch")
