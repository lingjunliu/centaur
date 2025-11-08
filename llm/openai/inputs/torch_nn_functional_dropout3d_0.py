
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def dropout3d_inputs():
    list_of_inputs = []
    
    input = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {"input": input, "p": 0.5, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32).numpy()
    input_dict = {"input": input, "p": 0.0, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 2, 3, 4, 5, dtype=torch.float64).numpy()
    input_dict = {"input": input, "p": 0.2, "training": False, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {"input": input, "p": 0.3, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 1, 2, 3, 3, dtype=torch.float16).numpy()
    input_dict = {"input": input, "p": 0.7, "training": True, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 3, 4, dtype=torch.float16).numpy()
    input_dict = {"input": input, "p": 0.9, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = (torch.randn(2, 4, 3, 3, 3, dtype=torch.float32) * 10).numpy()
    input_dict = {"input": input, "p": 0.4, "training": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.linspace(-50.0, 50.0, steps=1*2*2*3*4, dtype=torch.float64).reshape(1, 2, 2, 3, 4)
    input = x.numpy()
    input_dict = {"input": input, "p": 0.65, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros(2, 3, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {"input": input, "p": 0.25, "training": True, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 3, 3, dtype=torch.float64).numpy()
    input_dict = {"input": input, "p": 0.05, "training": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 8, 1, 8, 1, dtype=torch.float32).numpy()
    input_dict = {"input": input, "p": 0.8, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {"input": input, "p": 0.999, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.dropout3d"] = dropout3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.dropout3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout3d'.")


check_valid('torch.nn.functional.dropout3d', generated_inputs['torch.nn.functional.dropout3d'], lib="torch", suffix=0)
