
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def softmin_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2, 3)
    dim1 = 1
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(5)
    dim2 = 0
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(2, 3, 4)
    dim3 = 2
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1.0, 2.0], [3.0, 4.0]])
    dim4 = 1
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    dim5 = 0
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(1, 5, 5)
    dim6 = 1
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.randn(2, 2, 2, 2)
    dim7 = 3
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.Softmin"] = softmin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Softmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmin'.")

check_valid('torch.nn.Softmin', generated_inputs['torch.nn.Softmin'], lib="torch")
