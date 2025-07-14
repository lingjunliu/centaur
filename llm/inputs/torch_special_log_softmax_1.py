
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, dim=0, float32
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dim1 = 0
    dtype1 = torch.float32
    input_dict1 = {"input": input1, "dim": dim1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=1, float64
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dim2 = 1
    dtype2 = torch.float64
    input_dict2 = {"input": input2, "dim": dim2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, dim=2, float32, with negative values
    input3 = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float32)
    dim3 = 2
    dtype3 = torch.float32
    input_dict3 = {"input": input3, "dim": dim3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, dim=-1, float64
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dim4 = -1
    dtype4 = torch.float64
    input_dict4 = {"input": input4, "dim": dim4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor, dim=0, float32, all same values
    input5 = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    dim5 = 0
    dtype5 = torch.float32
    input_dict5 = {"input": input5, "dim": dim5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D tensor, dim=1, float64, large values
    input6 = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float64)
    dim6 = 1
    dtype6 = torch.float64
    input_dict6 = {"input": input6, "dim": dim6, "dtype": dtype6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 1D tensor, dim=0, float32, mixed positive and negative
    input7 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    dim7 = 0
    dtype7 = torch.float32
    input_dict7 = {"input": input7, "dim": dim7, "dtype": dtype7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D tensor, dim=0, float64, with zeros
    input8 = np.array([[[0.0, 2.0], [3.0, 0.0]], [[0.0, 6.0], [7.0, 0.0]]], dtype=np.float64)
    dim8 = 0
    dtype8 = torch.float64
    input_dict8 = {"input": input8, "dim": dim8, "dtype": dtype8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 2D tensor, dim=-1, float32
    input9 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dim9 = -1
    dtype9 = torch.float32
    input_dict9 = {"input": input9, "dim": dim9, "dtype": dtype9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 4D tensor, dim=3, float64
    input10 = np.random.rand(2, 3, 4, 5).astype(np.float64)
    dim10 = 3
    dtype10 = torch.float64
    input_dict10 = {"input": input10, "dim": dim10, "dtype": dtype10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.log_softmax_1"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.log_softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.log_softmax_1'.")

check_valid('torch.special.log_softmax', generated_inputs['torch.special.log_softmax_1'], lib="torch", suffix=1)
