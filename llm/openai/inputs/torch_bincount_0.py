
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def bincount_inputs():
    list_of_inputs = []

    input = torch.tensor([], dtype=torch.int64).numpy()
    weights = torch.tensor([], dtype=torch.float32).numpy()
    minlength = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([], dtype=torch.int64).numpy()
    weights = torch.tensor([], dtype=torch.float64).numpy()
    minlength = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([0], dtype=torch.int64).numpy()
    weights = torch.tensor([1.0], dtype=torch.float32).numpy()
    minlength = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    weights = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    minlength = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([1, 1, 2, 3, 3, 3], dtype=torch.int64).numpy()
    weights = torch.tensor([0.5, 1.5, 2.0, 1.0, 0.0, 3.0], dtype=torch.float64).numpy()
    minlength = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([0, 2, 4, 4, 7], dtype=torch.int64).numpy()
    weights = torch.tensor([1.0, 1.0, 1.0, 2.5, 3.5], dtype=torch.float32).numpy()
    minlength = 10
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([3, 3, 1, 0, 2], dtype=torch.int32).numpy()
    weights = torch.linspace(0, 1, steps=5, dtype=torch.float32).numpy()
    minlength = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([1, 1, 1, 2], dtype=torch.int64).numpy()
    weights = torch.tensor([-1.0, 2.0, -0.5, 3.0], dtype=torch.float64).numpy()
    minlength = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([5, 5, 5, 0], dtype=torch.int64).numpy()
    weights = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    minlength = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([100], dtype=torch.int64).numpy()
    weights = torch.tensor([1.0], dtype=torch.float32).numpy()
    minlength = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([0, 1, 1, 2], dtype=torch.int64).numpy()
    weights = torch.zeros(4, dtype=torch.float32).numpy()
    minlength = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    input = torch.tensor([2, 2, 2, 2, 0, 1, 4], dtype=torch.int64).numpy()
    weights = torch.tensor([10, 20, 30, 40, 50, 60, 70], dtype=torch.int64).numpy()
    minlength = 6
    list_of_inputs.append(copy.deepcopy({"input": input, "weights": weights, "minlength": minlength}))

    return list_of_inputs

generated_inputs["torch.bincount"] = bincount_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bincount'.")


check_valid('torch.bincount', generated_inputs['torch.bincount'], lib="torch", suffix=0)
