
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def imag_inputs():
    list_of_inputs = []
    
    input_val = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[[1+0j, 2+1j]], [[3+2j, 4+3j]]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([-1-2j, -3-4j, -5+6j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([1+0j, 2+0j, 3+0j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([0+1j, 0+2j, 0+3j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([5+7j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([1.5+2.5j, 3.7+4.2j], dtype=np.complex128)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[[[1+1j]]]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([1-2j, -3+4j, 5-6j, -7-8j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.imag"] = imag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.imag'.")


check_valid('torch.imag', generated_inputs['torch.imag'], lib="torch", suffix=0)
