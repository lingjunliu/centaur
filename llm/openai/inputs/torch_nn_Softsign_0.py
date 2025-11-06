
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softsign_inputs():
    list_of_inputs = []

    input = torch.tensor(3.5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.5, -2.5, 0.0],
                          [10.0, -0.1, 100.0]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-10, 10, steps=24, dtype=torch.float32).reshape(1, 3, 2, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randint(-10, 11, (5,), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-100, -1, 0, 1, 100]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0, 1e20, -1e20], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e-12, -1e-12, 1e6, -1e6], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    arr = np.arange(12, dtype=np.float32).reshape(3, 4).T
    input = arr
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[0.0, -0.5, 0.5]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 1, 3, 3, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Softsign"] = softsign_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softsign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softsign'.")


check_valid('torch.nn.Softsign', generated_inputs['torch.nn.Softsign'], lib="torch", suffix=0)
