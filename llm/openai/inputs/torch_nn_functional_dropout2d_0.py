
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dropout2d_inputs():
    list_of_inputs = []

    inp = torch.randn(2, 3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "p": 0.5, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "p": 0.0, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float64).reshape(1, 1, 4, 4).numpy()
    input_dict = {"input": inp, "p": 1.0, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 2, 16, 16, dtype=torch.float16).numpy()
    input_dict = {"input": inp, "p": 0.25, "training": True, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 4, 5, 5, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "p": 0.7, "training": False, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 3, 1, 1, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "p": 0.3, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (torch.randn(4, 2, 7, 9, dtype=torch.float32) * 5.0 - 2.5).numpy()
    input_dict = {"input": inp, "p": 0.9, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 8, 10, 5, dtype=torch.float64).numpy()
    input_dict = {"input": inp, "p": 0.05, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 10, 10, dtype=torch.float16).numpy()
    input_dict = {"input": inp, "p": 0.6, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 3, 10, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "p": 0.4, "training": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.full((6, 4, 6, 6), 2.0, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "p": 0.2, "training": True, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.zeros(3, 5, 12, 7, dtype=torch.float64).numpy()
    input_dict = {"input": inp, "p": 0.8, "training": True, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.dropout2d"] = dropout2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.dropout2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout2d'.")


check_valid('torch.nn.functional.dropout2d', generated_inputs['torch.nn.functional.dropout2d'], lib="torch", suffix=0)
