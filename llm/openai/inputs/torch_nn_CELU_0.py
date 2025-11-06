
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []
    
    input = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    input_dict = {"alpha": float(1.0), "inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, dtype=torch.float32).numpy()
    input_dict = {"alpha": float(0.5), "inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[ -2.0, -1.0, 0.0],
                           [ 0.5, 1.5, -0.5],
                           [ 2.0, -3.0, 3.0]]], dtype=torch.float64).numpy()
    input_dict = {"alpha": float(2.0), "inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-5, 5, steps=10, dtype=torch.float32).numpy()
    input_dict = {"alpha": float(0.01), "inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-1.0, -0.1, 0.0, 0.1, 1.0],
                          [2.0, -2.0, 3.0, -3.0, 4.0]], dtype=torch.float16).numpy()
    input_dict = {"alpha": float(3.5), "inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    input_dict = {"alpha": float(1.0), "inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 4, 4, dtype=torch.float32).numpy()
    input_dict = {"alpha": float(0.75), "inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.empty(0, 3, dtype=torch.float32).numpy()
    input_dict = {"alpha": float(2.25), "inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0], dtype=torch.float64).numpy()
    input_dict = {"alpha": float(5.0), "inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros(2, 2, dtype=torch.float32).numpy()
    input_dict = {"alpha": float(1.5), "inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.randn(5, 5, dtype=torch.float32).numpy()
    input = base[:, ::2]
    input_dict = {"alpha": float(0.3), "inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 1, 3, 2, 4, dtype=torch.float32).numpy()
    input_dict = {"alpha": float(2.5), "inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.CELU"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.CELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CELU'.")


check_valid('torch.nn.CELU', generated_inputs['torch.nn.CELU'], lib="torch", suffix=0)
