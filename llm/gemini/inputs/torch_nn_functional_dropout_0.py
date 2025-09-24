
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dropout_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p_value = 0.5
    training_value = True
    inplace_value = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.ones((2, 3), dtype=np.float32)
    p_value = 0.2
    training_value = False
    inplace_value = False  # Changed inplace to False to avoid issues
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(5, 5).astype(np.float32)
    p_value = 0.8
    training_value = True
    inplace_value = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    p_value = 0.3
    training_value = False
    inplace_value = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.zeros((4, 2, 2), dtype=np.float32)
    p_value = 0.7
    training_value = True
    inplace_value = False # changed inplace to False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.randn(10).astype(np.float32)
    p_value = 0.1
    training_value = True
    inplace_value = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.eye(3, dtype=np.float32)
    p_value = 0.9
    training_value = False
    inplace_value = False # Changed inplace to False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    p_value = 0.4
    training_value = True
    inplace_value = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.full((2, 2, 2), 5.0, dtype=np.float32)
    p_value = 0.6
    training_value = False
    inplace_value = False # Changed inplace to False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0.0], dtype=np.float32)
    p_value = 0.0
    training_value = True
    inplace_value = False
    input_dict = {"input": input_tensor, "p": p_value, "training": training_value, "inplace": inplace_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout'.")

check_valid('torch.nn.functional.dropout', generated_inputs['torch.nn.functional.dropout'], lib="torch", suffix=0)
