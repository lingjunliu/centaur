
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def float_power_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    exponent = 2.5
    out = torch.zeros((3,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((3, 4), dtype=torch.float64).numpy()
    exponent = 0.5
    out = torch.zeros((3, 4), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    exponent = -1.5
    out = torch.zeros((4,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3, 4), dtype=torch.float64).numpy()
    exponent = 3.0
    out = torch.zeros((2, 3, 4), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.5], dtype=torch.float64).numpy()
    exponent = 2.0
    out = torch.zeros((1,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((5,), dtype=torch.float64).numpy()
    exponent = 1.0
    out = torch.zeros((5,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([2.0, 3.0], dtype=torch.float64).numpy()
    exponent = -2.0
    out = torch.zeros((2,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((1, 2, 3), dtype=torch.float64).numpy()
    exponent = 0.75
    out = torch.zeros((1, 2, 3), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    exponent = 1.5
    out = torch.zeros((3,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((4,), dtype=torch.float64).numpy()
    exponent = -1.0
    out = torch.zeros((4,), dtype=torch.float64).numpy()

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
