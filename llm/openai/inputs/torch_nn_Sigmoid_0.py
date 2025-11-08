
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sigmoid_inputs():
    list_of_inputs = []

    input = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.random.randn(2, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.zeros((1, 2, 3, 4), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = np.arange(12, dtype=np.float32).reshape(4, 3)
    input = base.T
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = (np.arange(20, dtype=np.float32) - 10.0) / 2.0
    input = base[::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([np.nan, np.inf, -np.inf, -1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array(0.25, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.random.uniform(-5, 5, size=(2, 1, 3, 1, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.empty((2, 0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Sigmoid"] = sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Sigmoid'.")


check_valid('torch.nn.Sigmoid', generated_inputs['torch.nn.Sigmoid'], lib="torch", suffix=0)
