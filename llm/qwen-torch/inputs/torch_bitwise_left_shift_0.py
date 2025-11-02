
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def bitwise_left_shift_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()         # tensor
    other = torch.tensor([0, 1, 2], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([10, 20, 30], dtype=torch.int32).numpy()      # tensor
    other = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3), dtype=torch.int32).numpy()              # tensor
    other = torch.zeros((2, 3), dtype=torch.int32).numpy()            # tensor
    out = torch.zeros((2, 3), dtype=torch.int32).numpy()              # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([100, 200], dtype=torch.int32).numpy()        # tensor
    other = torch.tensor([2, 3], dtype=torch.int32).numpy()          # tensor
    out = torch.zeros((2,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0, 1, 2, 3], dtype=torch.int32).numpy()      # tensor
    other = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()     # tensor
    out = torch.zeros((4,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1, -2, -3], dtype=torch.int32).numpy()     # tensor
    other = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2], dtype=torch.int32).numpy()           # tensor
    other = torch.tensor([0, 1], dtype=torch.int32).numpy()          # tensor
    out = torch.zeros((2,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([10, 20], dtype=torch.int32).numpy()          # tensor
    other = torch.tensor([3, 4], dtype=torch.int32).numpy()          # tensor
    out = torch.zeros((2,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((3, 4), dtype=torch.int32).numpy()             # tensor
    other = torch.ones((3, 4), dtype=torch.int32).numpy()             # tensor
    out = torch.zeros((3, 4), dtype=torch.int32).numpy()              # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([5, 6, 7], dtype=torch.int32).numpy()         # tensor
    other = torch.tensor([2, 3, 4], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.bitwise_left_shift"] = bitwise_left_shift_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bitwise_left_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_left_shift'.")


check_valid('torch.bitwise_left_shift', generated_inputs['torch.bitwise_left_shift'], lib="torch", suffix=0)
