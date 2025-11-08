
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def float_power_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([2.0, 3.0, 4.0]).numpy()   # tensor
    exponent = 2.0 # float
    out = torch.zeros((3,)).numpy()    # tensor

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((3, 4)).numpy()
    exponent = 0.5
    out = torch.zeros((3, 4)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    exponent = 3.0
    out = torch.zeros((2, 2)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1.0]).numpy()
    exponent = -1.0
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((2, 2, 3)).numpy()
    exponent = 1.5
    out = torch.zeros((2, 2, 3)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.1, 0.2]).numpy()
    exponent = 0.0
    out = torch.zeros((2,)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 5)).numpy()
    exponent = 2.0
    out = torch.zeros((1, 5)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    exponent = -2.0
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((4, 2)).numpy()
    exponent = 0.3
    out = torch.zeros((4, 2)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([2.5, 3.5, 4.5, 5.5]).numpy()
    exponent = 1.0
    out = torch.zeros((4,)).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.float_power"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.float_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.float_power'.")


check_valid('torch.float_power', generated_inputs['torch.float_power'], lib="torch", suffix=0)
