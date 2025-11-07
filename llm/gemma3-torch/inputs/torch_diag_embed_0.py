
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def diag_embed_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3]).astype(np.int32)
    offset1 = 0
    dim1_1 = 0
    dim2_1 = 1
    input_dict1 = {
        "input": input1,
        "offset": offset1,
        "dim1": dim1_1,
        "dim2": dim2_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3, 4]).astype(np.float64)
    offset2 = 1
    dim1_2 = 0
    dim2_2 = 1
    input_dict2 = {
        "input": input2,
        "offset": offset2,
        "dim1": dim1_2,
        "dim2": dim2_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 2], [3, 4]]).astype(np.int64)
    offset3 = -1
    dim1_3 = 0
    dim2_3 = 1
    input_dict3 = {
        "input": input3,
        "offset": offset3,
        "dim1": dim1_3,
        "dim2": dim2_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    offset4 = 2
    dim1_4 = 1
    dim2_4 = 2
    input_dict4 = {
        "input": input4,
        "offset": offset4,
        "dim1": dim1_4,
        "dim2": dim2_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 2, 3, 4, 5]).astype(np.int16)
    offset5 = 0
    dim1_5 = 0
    dim2_5 = 1
    input_dict5 = {
        "input": input5,
        "offset": offset5,
        "dim1": dim1_5,
        "dim2": dim2_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]]).astype(np.complex128)
    offset6 = -2
    dim1_6 = 0
    dim2_6 = 1
    input_dict6 = {
        "input": input6,
        "offset": offset6,
        "dim1": dim1_6,
        "dim2": dim2_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.diag_embed"] = diag_embed_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diag_embed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diag_embed'.")


check_valid('torch.diag_embed', generated_inputs['torch.diag_embed'], lib="torch", suffix=0)
