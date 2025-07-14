
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def householder_product_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D input and h
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    h1 = np.array([0.5, 0.5], dtype=np.float64)
    input_dict1 = {"input": input1, "h": h1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values
    input2 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    h2 = np.array([0.5, 0.5], dtype=np.float32)
    input_dict2 = {"input": input2, "h": h2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Zeros
    input3 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    h3 = np.array([0.5, 0.5], dtype=np.float32)
    input_dict3 = {"input": input3, "h": h3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: h with zero values
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    h4 = np.array([0.0, 0.0], dtype=np.float32)
    input_dict4 = {"input": input4, "h": h4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large values
    input5 = np.array([[1e9, 2e9], [3e9, 4e9]], dtype=np.float32)
    h5 = np.array([0.5, 0.5], dtype=np.float32)
    input_dict5 = {"input": input5, "h": h5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different shape with same leading dim
    input6 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    h6 = np.array([0.5, 0.5], dtype=np.float32)
    input_dict6 = {"input": input6, "h": h6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Complex input
    input7 = np.array([[1.0+1j, 2.0-2j], [3.0+3j, 4.0-4j]], dtype=np.complex64)
    h7 = np.array([0.5+0.5j, 0.5-0.5j], dtype=np.complex64)
    input_dict7 = {"input": input7, "h": h7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Rectangular input, input.shape[-2] >= input.shape[-1]
    input8 = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=np.float32)
    h8 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict8 = {"input": input8, "h": h8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Identity Matrix
    input9 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    h9 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict9 = {"input": input9, "h": h9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Rectangular input, input.shape[-2] >= input.shape[-1]
    input10 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
    h10 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict10 = {"input": input10, "h": h10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.householder_product"] = householder_product_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.householder_product' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.householder_product'.")

check_valid('torch.linalg.householder_product', generated_inputs['torch.linalg.householder_product'], lib="torch", suffix=0)
