
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def log1p_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-0.5, 0.0, 0.5])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3))
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(5)
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.1, 0.2])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.9, 1.1])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-0.9, -0.1])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([2.0, 3.0])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.5, 2.5])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-0.2, 0.8])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log1p'.")


check_valid('torch.log1p', generated_inputs['torch.log1p'], lib="torch", suffix=0)
