
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def featurealphadropout_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(20, 16, 4, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.2,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(10, 8, 16, 16).numpy()
    input_dict = {
        "input": input,
        "p": 0.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(5, 32, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.7,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(1, 64, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.1,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(32, 1, 64, 64).numpy()
    input_dict = {
        "input": input,
        "p": 0.3,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(8, 128, 4, 4).numpy()
    input_dict = {
        "input": input,
        "p": 0.9,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(16, 32, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.4,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(128, 16, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.6,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(10, 32, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.2,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(64, 8, 16, 16).numpy()
    input_dict = {
        "input": input,
        "p": 0.8,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.FeatureAlphaDropout"] = featurealphadropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.FeatureAlphaDropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.FeatureAlphaDropout'.")


check_valid('torch.nn.FeatureAlphaDropout', generated_inputs['torch.nn.FeatureAlphaDropout'], lib="torch", suffix=0)
