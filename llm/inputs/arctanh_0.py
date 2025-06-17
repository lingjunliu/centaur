
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def arctanh_inputs():
    list_of_inputs = []

    input_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    input_dict = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_2 = np.array([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float32)
    input_dict = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_3 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_4 = np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32)
    input_dict = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_5 = np.array([0.0], dtype=np.float32)
    input_dict = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.arctanh"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arctanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arctanh'.")

check_valid('torch.arctanh', generated_inputs['torch.arctanh'], lib="torch")
