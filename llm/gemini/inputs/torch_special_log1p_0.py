
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def log1p_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = np.array([0.0, 1.0, 2.0, -0.5], dtype=np.float32)
    out1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor
    input2 = np.array([[0.0, 1.0], [-0.2, 0.5]], dtype=np.float64)
    out2 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    input3 = np.array([[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]], dtype=np.float32)
    out3 = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values
    input4 = np.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=np.float32)
    out4 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with large values
    input5 = np.array([1e-5, 1e-6, 1e-7], dtype=np.float64)
    out5 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with small values
    input6 = np.array([-1e-5, 1e-5, 0.0], dtype=np.float32)
    out6 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with mixed positive and negative values
    input7 = np.array([-0.75, 0.25, -0.1, 0.8], dtype=np.float64)
    out7 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Tensor with a single element
    input8 = np.array([0.5], dtype=np.float32)
    out8 = np.array([0.0], dtype=np.float32)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Tensor with zero
    input9 = np.array([0.0], dtype=np.float64)
    out9 = np.array([0.0], dtype=np.float64)
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Tensor with float16
    input10 = np.array([0.0, 0.5, -0.2], dtype=np.float16)
    out10 = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.log1p'.")

check_valid('torch.special.log1p', generated_inputs['torch.special.log1p'], lib="torch", suffix=0)
