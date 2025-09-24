
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def i0_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out1 = np.array([], dtype=np.float32)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, mixed positive and negative values
    input2 = np.array([[1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    out2 = np.array([], dtype=np.float64)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, all zeros
    input3 = np.zeros((2, 2, 2), dtype=np.float32)
    out3 = np.array([], dtype=np.float32)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, large positive values
    input4 = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    out4 = np.array([], dtype=np.float64)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor, small positive values
    input5 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    out5 = np.array([], dtype=np.float32)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Scalar value
    input6 = np.array(5.0, dtype=np.float32)
    out6 = np.array([], dtype=np.float32)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 3D tensor with positive values
    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    out7 = np.array([], dtype=np.float64)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 1D tensor, with zero value
    input8 = np.array([0.0], dtype=np.float32)
    out8 = np.array([], dtype=np.float32)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Empty tensor
    input9 = np.array([], dtype=np.float64)
    out9 = np.array([], dtype=np.float64)
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 4D tensor
    input10 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    out10 = np.array([], dtype=np.float32)
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.i0"] = i0_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i0'.")

check_valid('torch.special.i0', generated_inputs['torch.special.i0'], lib="torch", suffix=0)
