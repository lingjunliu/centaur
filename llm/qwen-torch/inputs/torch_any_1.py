
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def any_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0, 1, 2]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[False, True], [True, False]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0, 0, 0]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1, 0, 0]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[True, True], [False, False]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1, 0, 1]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[False, True], [True, False]], [[True, False], [False, True]]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.any_1"] = any_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.any_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_1'.")


check_valid('torch.any', generated_inputs['torch.any_1'], lib="torch", suffix=1)
