
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softmax_inputs():
    list_of_inputs = []

    input = np.array([-1.0, 0.0, 1.0, 2.0, -2.0], dtype=np.float32)
    dim = 0
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.linspace(-5, 5, 10).astype(np.float64)
    dim = -1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.array([[1.0, -1.0, 0.5], [3.2, 0.0, -2.1]], dtype=np.float32)
    dim = 1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    base = np.arange(12, dtype=np.float32).reshape(3, 4) - 6.0
    input = base.T
    dim = 0
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.random.randn(2, 3, 4).astype(np.float32)
    dim = 2
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.random.uniform(-2, 2, size=(5, 4, 3)).astype(np.float64)
    dim = -2
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.random.randn(2, 2, 3, 4).astype(np.float32)
    dim = 3
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.arange(15, dtype=np.float32).reshape(1, 3, 1, 5) - 7.5
    dim = 1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.random.randn(2, 1, 3, 1, 4).astype(np.float32)
    dim = -3
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]], dtype=np.float64)
    dim = 1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = np.array(
        [
            [[0.0, -np.inf, -1.0], [2.0, -np.inf, 3.0]],
            [[-np.inf, -np.inf, 0.0], [1.0, 2.0, -np.inf]],
        ],
        dtype=np.float32,
    )
    dim = 2
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    input = (np.random.randn(3, 4).astype(np.float32)) * 1e-3
    dim = -1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Softmax"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmax'.")


check_valid('torch.nn.Softmax', generated_inputs['torch.nn.Softmax'], lib="torch", suffix=0)
