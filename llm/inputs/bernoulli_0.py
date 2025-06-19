
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def bernoulli_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.tensor([0.1, 0.5, 0.9]).numpy()
    input_dict1 = {"input": input1, "generator": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor
    input2 = torch.rand(2, 3).numpy()
    input_dict2 = {"input": input2, "generator": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = torch.rand(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "generator": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor with some probabilities close to 0 and 1
    input4 = torch.tensor([0.001, 0.999, 0.2, 0.8]).numpy()
    input_dict4 = {"input": input4, "generator": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor with all probabilities equal to 0.5
    input5 = torch.full((4, 4), 0.5).numpy()
    input_dict5 = {"input": input5, "generator": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float tensor with different dtypes
    input6 = torch.rand(2, 3, dtype=torch.float64).numpy()
    input_dict6 = {"input": input6, "generator": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Float tensor with a generator
    generator = torch.Generator()
    input7 = torch.rand(2, 2).numpy()
    input_dict7 = {"input": input7, "generator": generator, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.bernoulli"] = bernoulli_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bernoulli' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bernoulli'.")

check_valid('torch.bernoulli', generated_inputs['torch.bernoulli'], lib="torch")
