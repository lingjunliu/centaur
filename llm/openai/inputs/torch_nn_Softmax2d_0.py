
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def softmax2d_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 12, 13, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(3, 10, 10, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.zeros(4, 1, 8, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.randn(1, 5, 3, 4, dtype=torch.float32) * 5 - 10).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.randn(2, 4, 5, 6, dtype=torch.float32) * 100).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 7, 2, 2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(1, 2, 32, 32, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    t = torch.randn(2, 3, 4, 5, dtype=torch.float32).transpose(1, 2)
    input = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    t = torch.randn(5, 6, 7, dtype=torch.float32).transpose(0, 1)
    input = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    x = torch.zeros(3, 4, 4, dtype=torch.float32)
    x[0] = -1.0
    x[1] = 0.0
    x[2] = 1.0
    input = x.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(7, 3, 1, 1, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.randn(2, 2, 3, 3, dtype=torch.float32) * 1e-6).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.rand(1, 4, 2, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Softmax2d"] = softmax2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softmax2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmax2d'.")


check_valid('torch.nn.Softmax2d', generated_inputs['torch.nn.Softmax2d'], lib="torch", suffix=0)
