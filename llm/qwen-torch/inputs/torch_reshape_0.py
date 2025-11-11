
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reshape_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.arange(4.).numpy()
    shape = (2, 2)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    shape = (-1,)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((3, 4)).numpy()
    shape = (2, 6)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((1, 5, 6)).numpy()
    shape = (30,)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.arange(12.).numpy()
    shape = (-1, 3)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((2, 2, 2)).numpy()
    shape = (4, -1)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.arange(6.).numpy()
    shape = (2, 3)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.zeros((4, 5)).numpy()
    shape = (-1, 2)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 3, 3)).numpy()
    shape = (1, 27)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.arange(8.).numpy()
    shape = (-1, 2, 2)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reshape'.")


check_valid('torch.reshape', generated_inputs['torch.reshape'], lib="torch", suffix=0)
