
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def logical_xor_inputs():
    list_of_inputs = []

    input = torch.tensor([True, False, True, False, True]).numpy()
    other = torch.tensor([False, False, True, True, False]).numpy()
    out = torch.empty(5, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[True, False, True],
                          [False, False, True]], dtype=torch.bool).numpy()
    other = torch.tensor([[False, True, False]], dtype=torch.bool).numpy()
    out = torch.empty((2, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-1, 0, 2, 0], dtype=torch.int8).numpy()
    other = torch.tensor([0, 0, 2, -3], dtype=torch.int8).numpy()
    out = torch.empty(4, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[0.0, 1.0],
                          [2.0, -0.0]], dtype=torch.float32).numpy()
    other = torch.tensor(0.0, dtype=torch.float32).numpy()
    out = torch.empty((2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(
        [[[True, False, True, False],
          [False, False, True, True],
          [True, True, False, False]],
         [[False, True, False, True],
          [True, True, False, False],
          [False, False, True, True]]],
        dtype=torch.bool
    ).numpy()
    other = torch.tensor([[[True], [False], [True]]], dtype=torch.bool).numpy()
    out = torch.empty((2, 3, 4), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([0, 1, 2], dtype=torch.uint8).numpy()
    other = torch.tensor([1, 0, 1], dtype=torch.uint8).numpy()
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[[0, 1, 0]],
                          [[2, 0, 3]]], dtype=torch.long).numpy()
    other = torch.tensor([[[0, 0, 1],
                           [1, 0, 0],
                           [0, 1, 0],
                           [1, 1, 1]]], dtype=torch.long).numpy()
    out = torch.empty((2, 4, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(True, dtype=torch.bool).numpy()
    other = torch.tensor([True, False, False, True], dtype=torch.bool).numpy()
    out = torch.empty(4, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[[[float('nan'), 0.0, 5.0]],
                           [[-1.0, 0.0, float('inf')]]]], dtype=torch.float64).numpy()
    other = torch.tensor([0.0, -2.0, 0.0], dtype=torch.float64).numpy()
    out = torch.empty((1, 2, 1, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty((0, 2), dtype=torch.bool).numpy()
    other = torch.tensor([[True, False]], dtype=torch.bool).numpy()
    out = torch.empty((0, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[True, False],
                          [False, True]], dtype=torch.bool).numpy()
    other = torch.tensor([[0, -1],
                          [2, 0]], dtype=torch.int32).numpy()
    out = torch.empty((2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[0.0],
                          [1.0],
                          [2.5],
                          [0.0],
                          [3.0]], dtype=torch.float16).numpy()
    other = torch.tensor([0.5], dtype=torch.float16).numpy()
    out = torch.empty((5, 1), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.logical_xor"] = logical_xor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logical_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_xor'.")


check_valid('torch.logical_xor', generated_inputs['torch.logical_xor'], lib="torch", suffix=0)
