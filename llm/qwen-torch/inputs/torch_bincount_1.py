
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def bincount_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0, 1, 2, 3, 4]).numpy()   # tensor
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).numpy() # tensor
    minlength = 5   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([0, 0, 1, 1, 2]).numpy()   # tensor
    weights = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]).numpy() # tensor
    minlength = 3   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([7, 6, 5, 4, 3]).numpy()   # tensor
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy() # tensor
    minlength = 8   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()   # tensor
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy() # tensor
    minlength = 6   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0]).numpy()   # tensor
    weights = torch.tensor([1.0]).numpy() # tensor
    minlength = 1   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([10, 10, 10]).numpy()   # tensor
    weights = torch.tensor([2.0, 3.0, 4.0]).numpy() # tensor
    minlength = 11   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0, 1, 2, 3, 4, 5]).numpy()   # tensor
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).numpy() # tensor
    minlength = 6   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([3, 3, 3, 3]).numpy()   # tensor
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy() # tensor
    minlength = 4   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0, 1, 2, 3, 4, 5, 6]).numpy()   # tensor
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).numpy() # tensor
    minlength = 7   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([2, 2, 2, 2, 2]).numpy()   # tensor
    weights = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]).numpy() # tensor
    minlength = 3   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.bincount_1"] = bincount_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bincount_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bincount_1'.")


check_valid('torch.bincount', generated_inputs['torch.bincount_1'], lib="torch", suffix=1)
