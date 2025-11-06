
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def alpha_dropout_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([0.5, -1.2, 3.3], dtype=torch.float32).numpy()
    p = 0.0
    training = False
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    p = 0.2
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.linspace(-5, 5, steps=60, dtype=torch.float64).reshape(3, 4, 5).numpy()
    p = 0.5
    training = True
    inplace = True
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float16).numpy()
    p = 0.9
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.empty((0, 3), dtype=torch.float32).numpy()
    p = 0.3
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.tensor(2.0, dtype=torch.float32).numpy()
    p = 0.1
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.randn(1, 2, 3, 4, 5, dtype=torch.float32).numpy()
    p = 1.0
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    p = 0.7
    training = True
    inplace = True
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([1e20, -1e-20, 3.14, -2.71], dtype=torch.float64).numpy()
    p = 0.6
    training = False
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0], dtype=torch.float32).numpy()
    p = 0.4
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.ones(1, dtype=torch.float32).numpy()
    p = 0.05
    training = False
    inplace = True
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.zeros((1, 3, 3), dtype=torch.float32).numpy()
    p = 0.25
    training = True
    inplace = False
    input_dict = {"input": input_arr, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.alpha_dropout"] = alpha_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.alpha_dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.alpha_dropout'.")


check_valid('torch.nn.functional.alpha_dropout', generated_inputs['torch.nn.functional.alpha_dropout'], lib="torch", suffix=0)
