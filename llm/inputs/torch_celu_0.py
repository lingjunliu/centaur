
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive values, alpha=1.0
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with negative values, alpha=1.0
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict2 = {"input": input2, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor with mixed values, alpha=1.0
    input3 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict3 = {"input": input3, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D tensor with positive values, alpha=0.5
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict4 = {"input": input4, "alpha": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor with negative values, alpha=0.5
    input5 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict5 = {"input": input5, "alpha": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D tensor with mixed values, alpha=0.5
    input6 = np.array([[-1.0, 0.0], [1.0, -2.0]], dtype=np.float32)
    input_dict6 = {"input": input6, "alpha": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 1D tensor with positive values, alpha=2.0
    input7 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict7 = {"input": input7, "alpha": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 1D tensor with negative values, alpha=2.0
    input8 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict8 = {"input": input8, "alpha": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 3D tensor with mixed values, alpha=1.5
    input9 = np.array([[[1.0, -1.0], [0.0, 2.0]], [[-2.0, 1.0], [3.0, 0.0]]], dtype=np.float32)
    input_dict9 = {"input": input9, "alpha": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 1D tensor with a different dtype
    input10 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict10 = {"input": input10, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.celu'.")

check_valid('torch.celu', generated_inputs['torch.celu'], lib="torch", suffix=0)
