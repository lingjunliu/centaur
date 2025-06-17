
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def arcsinh_inputs():
    list_of_inputs = []

    input1 = np.array([0, 1, 2, 3], dtype=np.float32)
    input_dict = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input2 = np.array([-0.5, 0.5, 1, 1.5], dtype=np.float64)
    input_dict = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input3 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input4 = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
    input_dict = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.arcsinh"] = arcsinh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arcsinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arcsinh'.")

check_valid('torch.arcsinh', generated_inputs['torch.arcsinh'], lib="torch")
