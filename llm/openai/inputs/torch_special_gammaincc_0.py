
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gammaincc_inputs():
    list_of_inputs = []

    input = torch.tensor([0.5, 1.0, 2.5], dtype=torch.float32).numpy()
    other = torch.tensor([0.1, 2.0, 5.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor(3.0, dtype=torch.float64).numpy()
    other = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[0.5], [5.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[0.2, 1.0, 10.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[1.0, 2.0, 3.0],
                          [0.1, 0.5, 1.5]], dtype=torch.float64).numpy()
    other = torch.tensor([[0.5, 1.5, 2.0],
                          [0.0, 0.2, 3.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[[0.5, 1.0, 2.0]],
                          [[3.0, 4.0, 5.0]]], dtype=torch.float32).numpy()
    other = torch.tensor([[[0.1],
                           [1.0],
                           [2.0],
                           [10.0]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[0.5, 1.0],
                          [2.0, 4.0]], dtype=torch.float16).numpy()
    other = torch.tensor([[0.0, 0.001],
                          [0.1, 10.0]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    other = torch.tensor([float('inf'), 0.0, 1.0, 100.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64).numpy()
    other = torch.tensor([1000.0, 5000.0, 10000.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[[[0.5, 1.0, 2.0]],
                           [[3.0, 4.0, 5.0]]]], dtype=torch.float32).numpy()
    other = torch.tensor([[[[0.1],
                            [1.0],
                            [5.0],
                            [20.0]]],
                          [[[0.2],
                            [2.0],
                            [10.0],
                            [50.0]]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([1e-6, 1e-3, 0.1, 0.5, 1.0], dtype=torch.float64).numpy()
    other = torch.tensor([0.0, 1e-12, 1e-4, 1e-2, 1e-1], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor(2.5, dtype=torch.float32).numpy()
    other = torch.tensor(3.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    input = torch.tensor([[[0.5, 1.0]],
                          [[2.0, 3.0]],
                          [[4.0, 5.0]]], dtype=torch.float64).numpy()
    other = torch.tensor([[[0.1],
                           [1.0]]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other}))

    return list_of_inputs

generated_inputs["torch.special.gammaincc"] = gammaincc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.gammaincc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.gammaincc'.")


check_valid('torch.special.gammaincc', generated_inputs['torch.special.gammaincc'], lib="torch", suffix=0)
