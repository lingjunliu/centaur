
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def expm1_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with positive values
    input1 = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with negative values
    input2 = np.array([-0.5, -1.0, -2.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor
    input3 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor
    input4 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with small values close to zero
    input5 = np.array([1e-6, -1e-6, 0.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with large values
    input6 = np.array([10.0, -10.0], dtype=np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with mixed positive and negative values
    input7 = np.array([-1.0, 0.5, 2.0, -0.2], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Tensor with zero values
    input8 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Tensor with different data type
    input9 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: 4D tensor
    input10 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Single element tensor
    input11 = np.array([0.5], dtype=np.float32)
    input_dict11 = {"input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: Empty tensor
    input12 = np.array([], dtype=np.float32)
    input_dict12 = {"input": input12}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.expm1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.expm1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.expm1'.")

check_valid('torch.special.expm1', generated_inputs['torch.special.expm1'], lib="torch", suffix=0)
