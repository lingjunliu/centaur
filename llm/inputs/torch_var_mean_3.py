
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def var_mean_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, unbiased=True, keepdim=False
    input1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dim1 = [0]
    unbiased1 = True
    keepdim1 = False

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "unbiased": unbiased1,
        "keepdim": keepdim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, unbiased=False, keepdim=True
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dim2 = [0]
    unbiased2 = False
    keepdim2 = True

    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "unbiased": unbiased2,
        "keepdim": keepdim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, unbiased=True, keepdim=False
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dim3 = [0, 1]
    unbiased3 = True
    keepdim3 = False

    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "unbiased": unbiased3,
        "keepdim": keepdim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor with negative values, unbiased=False, keepdim=True
    input4 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    dim4 = [0]
    unbiased4 = False
    keepdim4 = True

    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "unbiased": unbiased4,
        "keepdim": keepdim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor with dim=[0,1], unbiased=True, keepdim=True
    input5 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    dim5 = [0, 1]
    unbiased5 = True
    keepdim5 = True

    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "unbiased": unbiased5,
        "keepdim": keepdim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D tensor, unbiased=True, keepdim=True, empty dim list
    input6 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dim6 = []
    unbiased6 = True
    keepdim6 = True

    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "unbiased": unbiased6,
        "keepdim": keepdim6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 2D tensor, unbiased=False, keepdim=False, dim=[1]
    input7 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    dim7 = [1]
    unbiased7 = False
    keepdim7 = False

    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "unbiased": unbiased7,
        "keepdim": keepdim7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D tensor, unbiased=True, keepdim=True, dim=[1,2]
    input8 = np.random.rand(2, 3, 4).astype(np.float32)
    dim8 = [1, 2]
    unbiased8 = True
    keepdim8 = True

    input_dict8 = {
        "input": input8,
        "dim": dim8,
        "unbiased": unbiased8,
        "keepdim": keepdim8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 1D tensor with mixed positive and negative values, unbiased=False, keepdim=False
    input9 = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    dim9 = [0]
    unbiased9 = False
    keepdim9 = False

    input_dict9 = {
        "input": input9,
        "dim": dim9,
        "unbiased": unbiased9,
        "keepdim": keepdim9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 2D tensor, unbiased=True, keepdim=False, dim=[0]
    input10 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    dim10 = [0]
    unbiased10 = True
    keepdim10 = False

    input_dict10 = {
        "input": input10,
        "dim": dim10,
        "unbiased": unbiased10,
        "keepdim": keepdim10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.var_mean_3"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_3'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_3'], lib="torch", suffix=3)
