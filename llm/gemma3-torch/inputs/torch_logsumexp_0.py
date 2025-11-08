
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dim1 = 0
    keepdim1 = False
    out1 = np.array([], dtype=np.float32)
    
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dim2 = 1
    keepdim2 = True
    out2 = np.array([], dtype=np.float32)
    
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    dim3 = 0
    keepdim3 = False
    out3 = np.array([], dtype=np.float32)
    
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    dim4 = 2
    keepdim4 = True
    out4 = np.array([], dtype=np.float32)
    
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    dim5 = 1
    keepdim5 = False
    out5 = np.array([], dtype=np.float64)
    
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0], dtype=np.float16)
    dim6 = 0
    keepdim6 = True
    out6 = np.array([], dtype=np.float16)

    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "keepdim": keepdim6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    dim7 = (0, 1)
    keepdim7 = False
    out7 = np.array([], dtype=np.float32)

    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "keepdim": keepdim7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    dim8 = 0
    keepdim8 = True
    out8 = np.array([], dtype=np.float32)

    input_dict8 = {
        "input": input8,
        "dim": dim8,
        "keepdim": keepdim8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dim9 = 0
    keepdim9 = False
    out9 = np.array([], dtype=np.float32)

    input_dict9 = {
        "input": input9,
        "dim": dim9,
        "keepdim": keepdim9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dim10 = 1
    keepdim10 = True
    out10 = np.array([], dtype=np.float32)

    input_dict10 = {
        "input": input10,
        "dim": dim10,
        "keepdim": keepdim10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.logsumexp"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logsumexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp'.")


check_valid('torch.logsumexp', generated_inputs['torch.logsumexp'], lib="torch", suffix=0)
