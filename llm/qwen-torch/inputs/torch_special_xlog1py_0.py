
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def xlog1py_inputs():
    list_of_inputs = []
    
    # Input 1
    x = torch.tensor([0.0, 1.0, 2.0]).numpy()
    y = torch.tensor([1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    x = torch.tensor([0.5, 1.5, 2.5]).numpy()
    y = torch.tensor([0.5, 1.5, 2.5]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    x = torch.ones((3, 2)).numpy()
    y = torch.ones((3, 2)).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    x = torch.tensor([0.0]).numpy()
    y = torch.tensor([1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    x = torch.tensor([-1.0, -2.0]).numpy()
    y = torch.tensor([1.0, 1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    x = torch.tensor([0.0, 1.0, 2.0, 3.0]).numpy()
    y = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    x = torch.tensor([0.0, 1.0, 2.0]).numpy()
    y = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    x = torch.tensor([1.0, 2.0, 3.0]).numpy()
    y = torch.tensor([0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    x = torch.ones((2, 3)).numpy()
    y = torch.ones((2, 3)).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    x = torch.tensor([0.0, 1.0]).numpy()
    y = torch.tensor([0.5, 1.5]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py'.")


check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py'], lib="torch", suffix=0)
