
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def isnan_inputs():
    list_of_inputs = []

    input = np.array([1.0, float('nan'), 2.0])
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([[1.0, float('nan')], [2.0, 3.0]])
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([float('nan')] * 5)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([1 + 1j, float('nan') + 0j, 2 + 2j])
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([[1.0 + 1j, float('nan') + 0j], [2.0 + 0j, 3.0 + float('nan')*1j]])
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[[1.0, float('nan')], [2.0, 3.0]], [[4.0, 5.0], [float('nan'), 6.0]]])
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isnan"] = isnan_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isnan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isnan'.")

check_valid('torch.isnan', generated_inputs['torch.isnan'], lib="torch")
