
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def feature_alpha_dropout_inputs():
    list_of_inputs = []
    
    input_arr = torch.randn(2, 3, 2, 4, 4).numpy()
    input_dict = {"p": 0.0, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 1, 3, 3).numpy()
    input_dict = {"p": 1.0, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (-torch.rand(3, 4, 2, 5, 5)).numpy()
    input_dict = {"p": 0.2, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 2, 2, 3, 3, dtype=torch.float64).numpy()
    input_dict = {"p": 0.5, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 3, 6, 6).numpy()
    input_dict = {"p": 0.8, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 2, 4, 4, dtype=torch.float16).numpy()
    input_dict = {"p": 0.3, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 8, 2, 16, 16).numpy()
    input_dict = {"p": 0.7, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 3, 3, 3).numpy()
    input_dict = {"p": 1e-6, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(6, 1, 5, 5).numpy()
    input_dict = {"p": 0.9999, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 4, 2, 8, 8).numpy()
    input_dict = {"p": 0.4, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 1, 2, 6, 6).numpy()
    input_dict = {"p": 0.6, "inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 7, 7).numpy()
    input_dict = {"p": 0.15, "inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.FeatureAlphaDropout"] = feature_alpha_dropout_inputs()

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
