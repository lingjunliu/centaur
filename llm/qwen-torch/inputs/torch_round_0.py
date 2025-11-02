
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def round_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([4.7, -2.3, 9.1, -7.7])
    decimals = 0
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-0.5, 0.5, 1.5, 2.5])
    decimals = 0
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.1234567])
    decimals = 3
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1200.1234567])
    decimals = -3
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([10000])
    decimals = 3
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0])
    decimals = 0
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.5])
    decimals = 0
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.5])
    decimals = 0
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.234567])
    decimals = 2
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.234567])
    decimals = -1
    out = torch.zeros_like(input)

    input_dict = {
        "input": input,
        "decimals": decimals,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.round"] = round_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.round'.")


check_valid('torch.round', generated_inputs['torch.round'], lib="torch", suffix=0)
