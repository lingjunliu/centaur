
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def celu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    alpha = 1.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    alpha = 1.5
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    alpha = 2.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((1, 1, 2)).numpy()
    alpha = 0.5
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    alpha = 1.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([2.5, -1.5]).numpy()
    alpha = 3.0
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((5,)).numpy()
    alpha = 0.1
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-3.0, -2.0, -1.0]).numpy()
    alpha = 1.0
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 4, 5)).numpy()
    alpha = 2.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    alpha = 0.5
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.celu'.")


check_valid('torch.nn.functional.celu', generated_inputs['torch.nn.functional.celu'], lib="torch", suffix=0)
