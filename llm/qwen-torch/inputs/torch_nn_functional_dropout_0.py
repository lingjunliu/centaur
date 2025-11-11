
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dropout_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    p = 0.5
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    p = 0.3
    training = False
    inplace = True
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.zeros((1, 4, 5)).numpy()
    p = 0.8
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    p = 0.1
    training = False
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((3, 2)).numpy()
    p = 0.0
    training = True
    inplace = True
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((4, 3, 2)).numpy()
    p = 0.9
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, -2.0], [3.0, 4.0]]).numpy()
    p = 0.6
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((1, 1)).numpy()
    p = 0.2
    training = False
    inplace = True
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((2, 2, 2, 2)).numpy()
    p = 0.4
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([5.0, 6.0, 7.0]).numpy()
    p = 0.7
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout'.")


check_valid('torch.nn.functional.dropout', generated_inputs['torch.nn.functional.dropout'], lib="torch", suffix=0)
