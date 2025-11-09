
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def isreal_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1+1j, 2+2j, 3+3j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1+0j, 2+0j, 3+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1, 1+1j, 2+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1+1j, 2+2j], [3+0j, 4+0j]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1+1j, -2-2j, -3+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(5.0).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(5.0+0j).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0+0j, 0+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.isreal"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isreal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isreal'.")


check_valid('torch.isreal', generated_inputs['torch.isreal'], lib="torch", suffix=0)
