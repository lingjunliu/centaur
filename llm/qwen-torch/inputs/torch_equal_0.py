
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def equal_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((3, 4)).numpy()
    other = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((2, 3, 4)).numpy()
    other = torch.zeros((2, 3, 4)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    other = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    other = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.equal"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.equal'.")


check_valid('torch.equal', generated_inputs['torch.equal'], lib="torch", suffix=0)
