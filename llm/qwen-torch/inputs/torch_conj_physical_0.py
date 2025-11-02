
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def conj_physical_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64).numpy()  # tensor
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3), dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1-2j, 3-4j, 5-6j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((3, 4), dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0+2j, 3.0+4j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 2, 3), dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0-2j, 3.0-4j, 5.0-6j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0+2j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
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
