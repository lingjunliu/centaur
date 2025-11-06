
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def i0e_inputs():
    list_of_inputs = []

    input = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.1, 2.0, -4.0], [10.0, -0.5, 0.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-2, 2, steps=24, dtype=torch.float64).reshape(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(5.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([50.0, 100.0, 300.0, -50.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(0, 5, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0, -1, 2], [3, -4, 5]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[-1, 2], [3, -4]], [[5, -6], [7, -8]]], dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 1.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.special.i0e"] = i0e_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i0e'.")


check_valid('torch.special.i0e', generated_inputs['torch.special.i0e'], lib="torch", suffix=0)
