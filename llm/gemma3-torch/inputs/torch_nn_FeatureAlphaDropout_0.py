
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def feature_alpha_dropout_inputs():
    list_of_inputs = []

    input1 = np.random.rand(20, 16, 4, 32, 32).astype(np.float32)
    input_dict1 = {"p": 0.2, "inplace": False, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(10, 8, 2, 16, 16).astype(np.float32)
    input_dict2 = {"p": 0.5, "inplace": True, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(5, 32, 1, 8, 8).astype(np.float32)
    input_dict3 = {"p": 0.1, "inplace": False, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 64, 8, 4, 4).astype(np.float32)
    input_dict4 = {"p": 0.9, "inplace": True, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(32, 16, 2, 4, 4).astype(np.float32)
    input_dict5 = {"p": 0.0, "inplace": False, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(16, 3, 1, 1).astype(np.float32)
    input_dict6 = {"p": 0.7, "inplace": True, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(8, 64, 4, 4).astype(np.float32)
    input_dict7 = {"p": 0.3, "inplace": False, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(4, 128, 2, 2).astype(np.float32)
    input_dict8 = {"p": 1.0, "inplace": True, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(64, 3, 16, 16).astype(np.float32)
    input_dict9 = {"p": 0.6, "inplace": False, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 1, 32, 32).astype(np.float32)
    input_dict10 = {"p": 0.8, "inplace": True, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
