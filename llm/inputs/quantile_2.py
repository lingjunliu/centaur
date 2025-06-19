
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def quantile_inputs():
    list_of_inputs = []

    input1 = np.random.randn(5).astype(np.float32)
    q1 = 0.5
    dim1 = None
    keepdim1 = False
    interpolation1 = 'linear'

    input_dict1 = {
        "input": input1,
        "q": q1,
        "dim": dim1,
        "keepdim": keepdim1,
        "interpolation": interpolation1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.randn(2, 3).astype(np.float32)
    q2 = np.array([0.25, 0.5, 0.75]).astype(np.float32)
    dim2 = 1
    keepdim2 = True
    interpolation2 = 'linear'

    input_dict2 = {
        "input": input2,
        "q": q2,
        "dim": dim2,
        "keepdim": keepdim2,
        "interpolation": interpolation2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.arange(4).astype(np.float64)
    q3 = 0.6
    dim3 = None
    keepdim3 = False
    interpolation3 = 'lower'

    input_dict3 = {
        "input": input3,
        "q": q3,
        "dim": dim3,
        "keepdim": keepdim3,
        "interpolation": interpolation3,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.0, 0.0], [1.0, 2.0]]).astype(np.float32)
    q4 = 0.3
    dim4 = 0
    keepdim4 = False
    interpolation4 = 'higher'

    input_dict4 = {
        "input": input4,
        "q": q4,
        "dim": dim4,
        "keepdim": keepdim4,
        "interpolation": interpolation4,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(2, 2, 2).astype(np.float32)
    q5 = np.array([0.1, 0.9]).astype(np.float32)
    dim5 = 1
    keepdim5 = False
    interpolation5 = 'nearest'

    input_dict5 = {
        "input": input5,
        "q": q5,
        "dim": dim5,
        "keepdim": keepdim5,
        "interpolation": interpolation5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 3, 4, 5]).astype(np.float64)
    q6 = 0.5
    dim6 = None
    keepdim6 = False
    interpolation6 = 'midpoint'

    input_dict6 = {
        "input": input6,
        "q": q6,
        "dim": dim6,
        "keepdim": keepdim6,
        "interpolation": interpolation6,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    q7 = np.array([0.25, 0.75]).astype(np.float32)
    dim7 = 0
    keepdim7 = True
    interpolation7 = 'linear'

    input_dict7 = {
        "input": input7,
        "q": q7,
        "dim": dim7,
        "keepdim": keepdim7,
        "interpolation": interpolation7,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.quantile_2"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.quantile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantile_2'.")

check_valid('torch.quantile', generated_inputs['torch.quantile_2'], lib="torch")
