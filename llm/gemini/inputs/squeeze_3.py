
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_squeeze_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 1, 3, 1, 4).numpy()
    dim1 = (1, 3)
    input_dict1 = {
        "input": input1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 5, 1, 7).numpy()
    dim2 = (0, 2)
    input_dict2 = {
        "input": input2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 2, 1, 4, 1).numpy()
    dim3 = (2, 4)
    input_dict3 = {
        "input": input3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 1, 1).numpy()
    dim4 = (0, 1, 2, 3)
    input_dict4 = {
        "input": input4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = ()
    input_dict5 = {
        "input": input5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 4, 5).numpy()
    dim6 = (0,)
    input_dict6 = {
        "input": input6,
        "dim": dim6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 1, 4, 1).numpy()
    dim7 = (1,)
    input_dict7 = {
        "input": input7,
        "dim": dim7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(1, 2, 1, 3).numpy()
    dim8 = (0, 2)
    input_dict8 = {
        "input": input8,
        "dim": dim8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randint(0, 10, (2, 1, 3, 1)).numpy()
    dim9 = (1, 3)
    input_dict9 = {
        "input": input9,
        "dim": dim9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 5, 1, 7).numpy()
    dim10 = (0,)
    input_dict10 = {
        "input": input10,
        "dim": dim10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.squeeze_3"] = torch_squeeze_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.squeeze_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.squeeze_3'.")

check_valid('torch.squeeze', generated_inputs['torch.squeeze_3'], lib="torch")
