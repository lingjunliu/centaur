
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bincount_inputs():
    list_of_inputs = []
    
    input1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    weights1 = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
    minlength1 = 0
    
    input_dict1 = {
        "input": input1,
        "weights": weights1,
        "minlength": minlength1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    weights2 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    minlength2 = 3
    
    input_dict2 = {
        "input": input2,
        "weights": weights2,
        "minlength": minlength2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([], dtype=np.int64)
    weights3 = None
    minlength3 = 5
    
    input_dict3 = {
        "input": input3,
        "weights": weights3,
        "minlength": minlength3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    weights4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    minlength4 = 0
    
    input_dict4 = {
        "input": input4,
        "weights": weights4,
        "minlength": minlength4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([0, 1, 0, 2, 1, 0], dtype=np.int64)
    weights5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    minlength5 = 3
    
    input_dict5 = {
        "input": input5,
        "weights": weights5,
        "minlength": minlength5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([7, 7, 7, 7], dtype=np.int64)
    weights6 = np.array([1, 1, 1, 1], dtype=np.float32)
    minlength6 = 8
    
    input_dict6 = {
        "input": input6,
        "weights": weights6,
        "minlength": minlength6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    weights7 = np.array([1, 2, 1, 2, 1, 2], dtype=np.float32)
    minlength7 = 0

    input_dict7 = {
        "input": input7,
        "weights": weights7,
        "minlength": minlength7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([10], dtype=np.int64)
    weights8 = np.array([1.0], dtype=np.float32)
    minlength8 = 5
    
    input_dict8 = {
        "input": input8,
        "weights": weights8,
        "minlength": minlength8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    weights9 = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    minlength9 = 10
    
    input_dict9 = {
        "input": input9,
        "weights": weights9,
        "minlength": minlength9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    weights10 = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    minlength10 = 2

    input_dict10 = {
        "input": input10,
        "weights": weights10,
        "minlength": minlength10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.bincount"] = bincount_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bincount'.")


check_valid('torch.bincount', generated_inputs['torch.bincount'], lib="torch", suffix=0)
