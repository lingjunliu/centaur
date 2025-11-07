
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def amin_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dim1 = 0
    keepdim1 = True
    out1 = np.empty((1, 3), dtype=np.float64)

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float64)
    dim2 = 1
    keepdim2 = False
    out2 = np.empty((2,), dtype=np.float64)

    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(3, 4, 5).astype(np.float64)
    dim3 = (0, 1)
    keepdim3 = True
    out3 = np.empty((1, 1, 5), dtype=np.float64)

    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.5, 0.2], [0.8, 0.1]], dtype=np.float64)
    dim4 = None
    keepdim4 = False
    out4 = np.empty(() , dtype=np.float64)

    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([10], dtype=np.float64)
    dim5 = 0
    keepdim5 = True
    out5 = np.empty((1,), dtype=np.float64)

    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    dim6 = 0
    keepdim6 = False
    out6 = np.empty((2,), dtype=np.float64)

    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "keepdim": keepdim6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(2, 2, 2, 2).astype(np.float64)
    dim7 = (1, 3)
    keepdim7 = False
    out7 = np.empty((2, 2), dtype=np.float64)

    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "keepdim": keepdim7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs["torch.amin"] = amin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.amin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin'.")


check_valid('torch.amin', generated_inputs['torch.amin'], lib="torch", suffix=0)
