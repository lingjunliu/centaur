
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def lcm_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([5, 10, 15]).numpy()   # tensor
    other = torch.tensor([3, 4, 5]).numpy()     # tensor
    out = torch.tensor([15, 20, 15]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    other = torch.tensor([4, 5, 6]).numpy()   # tensor
    out = torch.tensor([4, 10, 6]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0, 1, 2]).numpy()   # tensor
    other = torch.tensor([0, 2, 3]).numpy()   # tensor
    out = torch.tensor([0, 2, 6]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([10, 20]).numpy()   # tensor
    other = torch.tensor([5, 10]).numpy()   # tensor
    out = torch.tensor([10, 20]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([7, 11, 13]).numpy()   # tensor
    other = torch.tensor([2, 3, 5]).numpy()   # tensor
    out = torch.tensor([14, 33, 65]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([100]).numpy()   # tensor
    other = torch.tensor([50]).numpy()   # tensor
    out = torch.tensor([100]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-5, -10, -15]).numpy()   # tensor
    other = torch.tensor([3, 4, 5]).numpy()   # tensor
    out = torch.tensor([15, 20, 15]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([2, 4, 6]).numpy()   # tensor
    other = torch.tensor([-3, -6, -9]).numpy()   # tensor
    out = torch.tensor([6, 12, 18]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    other = torch.tensor([0, 1, 2]).numpy()   # tensor
    out = torch.tensor([0, 2, 6]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([3, 5, 7]).numpy()   # tensor
    other = torch.tensor([2, 4, 6]).numpy()   # tensor
    out = torch.tensor([6, 20, 42]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.lcm"] = lcm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lcm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lcm'.")


check_valid('torch.lcm', generated_inputs['torch.lcm'], lib="torch", suffix=0)
