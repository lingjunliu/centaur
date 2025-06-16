
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hardshrink_inputs():
    list_of_inputs = []

    input_1 = torch.randn(2).numpy()
    lambd_1 = 0.5
    input_dict_1 = {"lambd": lambd_1, "input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randn(2, 3).numpy()
    lambd_2 = 0.7
    input_dict_2 = {"lambd": lambd_2, "input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randn(2, 3, 4).numpy()
    lambd_3 = 0.3
    input_dict_3 = {"lambd": lambd_3, "input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randn(1, 5, 5, 5).numpy()
    lambd_4 = 0.9
    input_dict_4 = {"lambd": lambd_4, "input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = torch.randn(size=(10,)).numpy()
    lambd_5 = 0.1
    input_dict_5 = {"lambd": lambd_5, "input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    input_6 = torch.randn(3, 3).float().numpy()
    lambd_6 = 1.2
    input_dict_6 = {"lambd": lambd_6, "input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Hardshrink"] = hardshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.Hardshrink', generated_inputs['torch.nn.Hardshrink'], lib="torch")
