
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conj_physical_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D complex tensor
    input_tensor = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D complex tensor
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Real tensor (float32)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Real tensor (float64)
    input_tensor = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex tensor with negative values
    input_tensor = np.array([-1+2j, -3-4j, 5-6j], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D complex tensor
    input_tensor = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex128)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element complex tensor
    input_tensor = np.array([3+4j], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex tensor with zeros
    input_tensor = np.array([0+0j, 1+0j, 0+1j], dtype=np.complex128)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Integer tensor
    input_tensor = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D complex tensor
    input_tensor = np.array([[[[1+1j]]]], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Large 2D complex tensor
    input_tensor = np.random.randn(5, 5).astype(np.float32) + 1j * np.random.randn(5, 5).astype(np.float32)
    input_tensor = input_tensor.astype(np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.conj_physical"] = conj_physical_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.conj_physical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.conj_physical'.")


check_valid('torch.conj_physical', generated_inputs['torch.conj_physical'], lib="torch", suffix=0)
