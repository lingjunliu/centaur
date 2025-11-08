
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def tanh_inputs():
    list_of_inputs = []

    input = torch.randn(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-3.0, 0.0, 3.0],
                          [10.0, -10.0, 1.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-3.5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=torch.float16).reshape(1, 1, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(-6., 6., dtype=torch.float32).reshape(3, 4).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+2j, -3+0.5j, 0+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    real = torch.randn(2, 3, dtype=torch.float64)
    imag = torch.randn(2, 3, dtype=torch.float64)
    input = (real + 1j * imag).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('inf'), float('-inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([np.nan, -2.5, 2.5], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-5, 5, steps=11, dtype=torch.float32)[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(2, 0, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e-12, -1e-12, 1e-30, -1e-30], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Tanh"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Tanh'.")


check_valid('torch.nn.Tanh', generated_inputs['torch.nn.Tanh'], lib="torch", suffix=0)
