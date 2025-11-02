
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def imag_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1+2j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1-2j, -3-4j, -5-6j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0+0j, 1+1j], [2+2j, 3+3j]], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.5+0.5j, 1.5+1.5j, 2.5+2.5j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.0+0.0j, 1.0+1.0j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1j, 2j, 3j, 4j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1j, -2j, -3j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0+1j, 0+2j, 0+3j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
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
