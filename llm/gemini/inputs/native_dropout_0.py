
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def native_dropout_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4).astype(np.float32)
    p1 = 0.5
    training1 = True
    input_dict1 = {"input": input1, "p": p1, "training": training1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 3, 4).astype(np.float64)
    p2 = 0.2
    training2 = False
    input_dict2 = {"input": input2, "p": p2, "training": training2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(1, 5, 5, 2).astype(np.float32)
    p3 = 0.8
    training3 = True
    input_dict3 = {"input": input3, "p": p3, "training": training3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(4).astype(np.float64)
    p4 = 0.3
    training4 = False
    input_dict4 = {"input": input4, "p": p4, "training": training4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    p5 = 0.6
    training5 = True
    input_dict5 = {"input": input5, "p": p5, "training": training5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.native_dropout"] = native_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.native_dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.native_dropout'.")

check_valid('torch.native_dropout', generated_inputs['torch.native_dropout'], lib="torch")
