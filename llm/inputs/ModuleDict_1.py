
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import copy
import numpy as np

def torch_nn_ModuleDict_inputs():
    list_of_inputs = []

    # Input 1: Empty ModuleDict
    modules = {}
    input_dict = {"modules": modules}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: ModuleDict with one module
    modules = {"conv1": nn.Conv2d(1, 1, 3)}
    input_dict = {"modules": modules}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: ModuleDict with multiple modules
    modules = {"linear": nn.Linear(10, 20), "relu": nn.ReLU(), "dropout": nn.Dropout(0.5)}
    input_dict = {"modules": modules}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: ModuleDict initialized with a list of tuples
    modules = [("bn", nn.BatchNorm1d(10)), ("sigmoid", nn.Sigmoid())]
    new_modules = {}
    for key, value in modules:
        new_modules[key] = value
    modules = new_modules
    input_dict = {"modules": modules}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: ModuleDict with different types of modules
    modules = {"embedding": nn.Embedding(100, 32), "lstm": nn.LSTM(32, 64), "classifier": nn.Linear(64, 10)}
    input_dict = {"modules": modules}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ModuleDict_1"] = torch_nn_ModuleDict_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ModuleDict_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ModuleDict_1'.")

check_valid('torch.nn.ModuleDict', generated_inputs['torch.nn.ModuleDict_1'], lib="torch", suffix=1)
