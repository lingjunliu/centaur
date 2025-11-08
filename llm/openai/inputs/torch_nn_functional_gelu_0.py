
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gelu_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[0.1, -0.2, 3.5],
                              [-4.0, 2.2, 0.0]], dtype=torch.float64).numpy()
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "tanh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn((2, 2, 3), dtype=torch.float16).numpy()
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-10, 10, steps=48, dtype=torch.float32).reshape(2, 3, 2, 4).numpy()
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "tanh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor(0.5, dtype=torch.float32).numpy()
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([], dtype=np.float32)
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "tanh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input_arr = base[:, ::2]
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([-1e-8, 0.0, 1e-8], dtype=np.float64)
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "tanh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([-20.0, -5.0, 5.0, 20.0], dtype=np.float16)
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.rand((2, 3, 1, 4), dtype=torch.float32).numpy()
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "tanh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn((1, 2, 3, 4, 5), dtype=torch.float32).numpy()
    input_dict = {"input": copy.deepcopy(input_arr), "approximate": "tanh"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.gelu"] = gelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.gelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.gelu'.")


check_valid('torch.nn.functional.gelu', generated_inputs['torch.nn.functional.gelu'], lib="torch", suffix=0)
