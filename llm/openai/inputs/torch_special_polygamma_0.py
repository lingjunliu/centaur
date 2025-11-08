
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def polygamma_inputs():
    list_of_inputs = []

    n = 0
    input = torch.tensor([0.5, 1.0, 2.5, 10.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 1
    input = torch.tensor([[-0.5, 0.0, 0.1, 1.5],
                          [2.0, -1.2, 3.3, 4.4]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 2
    input = np.array(3.14159265, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 3
    input = torch.tensor([[[-3.0, -0.5, 0.5]],
                          [[1.0, 2.0, 5.0]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 5
    input = np.array([-3.0, -2.0, -1.0, 0.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 10
    input = np.array([50.0, 100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 0
    input = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 1
    input = np.array([-np.inf, -1.0, np.nan, np.inf, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 4
    base = np.linspace(-2.0, 2.0, num=16, dtype=np.float32)
    input = base.reshape(2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 2
    input = np.array([[1e-6, -1e-6, 1e-3],
                      [-1e-3, 2e-2, -2e-2],
                      [0.5, -0.25, 0.25]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 7
    input = np.arange(-6.0, 6.0, dtype=np.float64).reshape(4, 3)[:, ::2]
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    n = 8
    input = np.array(-0.75, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "input": input}))

    return list_of_inputs

generated_inputs["torch.special.polygamma"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.polygamma'.")


check_valid('torch.special.polygamma', generated_inputs['torch.special.polygamma'], lib="torch", suffix=0)
