
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def calculate_gain_inputs():
    list_of_inputs = []

    # Input 1: ReLU
    input_dict = {"nonlinearity": "relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: sigmoid
    input_dict = {"nonlinearity": "sigmoid"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: tanh
    input_dict = {"nonlinearity": "tanh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: linear
    input_dict = {"nonlinearity": "linear"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: leaky_relu with negative slope 0.01 (default)
    input_dict = {"nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: leaky_relu with negative slope 0.2
    input_dict = {"nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: leaky_relu with negative slope 0.5
    input_dict = {"nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: leaky_relu with negative slope 0.9
    input_dict = {"nonlinearity": "leaky_relu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: selu
    input_dict = {"nonlinearity": "selu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.calculate_gain"] = calculate_gain_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.calculate_gain' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.calculate_gain'.")

check_valid('torch.nn.init.calculate_gain', generated_inputs['torch.nn.init.calculate_gain'], lib="torch", suffix=0)
