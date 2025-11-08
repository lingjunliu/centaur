
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def combinations_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    r = 2
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    r = 3
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    r = 1
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([-5.5, 0.0, 5.5], dtype=torch.float64).numpy()
    r = 2
    with_replacement = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([True, False, True, False], dtype=torch.bool).numpy()
    r = 2
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([42], dtype=torch.int64).numpy()
    r = 1
    with_replacement = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([10, 20], dtype=torch.int64).numpy()
    r = 3
    with_replacement = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([-3, -2, -1, 0], dtype=torch.int64).numpy()
    r = 4
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([1, 1, 2, 2], dtype=torch.int64).numpy()
    r = 2
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64).numpy()
    r = 2
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.arange(6, dtype=torch.int64).numpy()
    r = 3
    with_replacement = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    input_arr = torch.tensor([10_000_000_000, -10_000_000_000, 0], dtype=torch.int64).numpy()
    r = 2
    with_replacement = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "r": r, "with_replacement": with_replacement}))

    return list_of_inputs

generated_inputs["torch.combinations"] = combinations_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.combinations' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.combinations'.")


check_valid('torch.combinations', generated_inputs['torch.combinations'], lib="torch", suffix=0)
