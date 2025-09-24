
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def feature_alpha_dropout_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    p_value = 0.5
    training_flag = True
    inplace_flag = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(1, 5, 5).astype(np.float32)
    p_value = 0.2
    training_flag = False
    inplace_flag = True
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(4, 4, 4, 4).astype(np.float32)
    p_value = 0.1
    training_flag = False
    inplace_flag = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randn(2, 2).astype(np.float32)
    p_value = 0.9
    training_flag = True
    inplace_flag = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.zeros((3, 3)).astype(np.float32)
    p_value = 0.5
    training_flag = True
    inplace_flag = True
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.ones((1, 1)).astype(np.float32)
    p_value = 0.3
    training_flag = False
    inplace_flag = True
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.randn(1, 1, 1, 1, 1).astype(np.float32)
    p_value = 0.7
    training_flag = True
    inplace_flag = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    p_value = 1.0
    training_flag = False
    inplace_flag = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.random.randn(3, 5).astype(np.float32)
    p_value = 0.6
    training_flag = True
    inplace_flag = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.random.randn(2, 2, 2).astype(np.float32)
    p_value = 0.4
    training_flag = False
    inplace_flag = True
    input_dict = {"input": input_tensor, "p": p_value, "training": training_flag, "inplace": inplace_flag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.feature_alpha_dropout"] = feature_alpha_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.feature_alpha_dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.feature_alpha_dropout'.")

check_valid('torch.nn.functional.feature_alpha_dropout', generated_inputs['torch.nn.functional.feature_alpha_dropout'], lib="torch", suffix=0)
