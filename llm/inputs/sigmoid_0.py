
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_sigmoid_inputs():
    list_of_inputs = []

    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    input_3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    input_4 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = np.array([0.0], dtype=np.float32)
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sigmoid"] = torch_sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sigmoid'.")

check_valid('torch.sigmoid', generated_inputs['torch.sigmoid'], lib="torch")
