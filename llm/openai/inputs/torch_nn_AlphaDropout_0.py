
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def alpha_dropout_inputs():
    list_of_inputs = []

    inp = torch.randn(20, 16, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.2, "inplace": False, "input": inp}))

    inp = torch.linspace(-1, 1, steps=10, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.5, "inplace": True, "input": inp}))

    inp = torch.randn(2, 3, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.0, "inplace": False, "input": inp}))

    inp = torch.randn(1, 3, 8, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.9, "inplace": True, "input": inp}))

    inp = torch.tensor(1.2345, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 1.0, "inplace": False, "input": inp}))

    inp = torch.randn(2, 2, 2, 2, 2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.05, "inplace": False, "input": inp}))

    inp = torch.randn(64, 128, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.3, "inplace": True, "input": inp}))

    inp = torch.randn(16, 16, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.4, "inplace": False, "input": inp}))

    inp = torch.zeros(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.7, "inplace": True, "input": inp}))

    inp = torch.linspace(-3, 3, steps=12, dtype=torch.float32).reshape(3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.15, "inplace": False, "input": inp}))

    inp = (torch.randn(4, 4, dtype=torch.float32) * 1000.0).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.6, "inplace": True, "input": inp}))

    inp = torch.empty(0, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"p": 0.25, "inplace": False, "input": inp}))

    return list_of_inputs

generated_inputs["torch.nn.AlphaDropout"] = alpha_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AlphaDropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AlphaDropout'.")


check_valid('torch.nn.AlphaDropout', generated_inputs['torch.nn.AlphaDropout'], lib="torch", suffix=0)
