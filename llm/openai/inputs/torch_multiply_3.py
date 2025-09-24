
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multiply_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([1.0, -2.5, 3.0], dtype=torch.float32).numpy()
    other = np.int64(2)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.tensor([[1.0, 2.0], [-3.5, 4.0]], dtype=torch.float64).numpy()
    other = np.int32(-3)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.tensor(
        [[[1, -2, 3], [4, 5, -6]],
         [[-7, 8, 9], [10, -11, 12]]],
        dtype=torch.int16
    ).numpy()
    other = np.int8(5)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.arange(24, dtype=torch.int64).reshape(1, 2, 3, 4).numpy()
    other = np.int64(0)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.tensor(3.14, dtype=torch.float32).numpy()
    other = np.int16(4)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.empty((0, 5), dtype=torch.float64).numpy()
    other = np.int32(7)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    base = torch.arange(10, dtype=torch.float32).numpy()
    input = base[::2]
    other = np.int8(-1)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.tensor([[0, 128, 255], [10, 20, 30]], dtype=torch.uint8).numpy()
    other = np.int8(2)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = torch.tensor([1 + 2j, -3 + 4j, 0 - 1j], dtype=torch.complex64).numpy()
    other = np.int16(3)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.randn((1, 2, 1, 3, 2), dtype=torch.float16).numpy()
    other = np.int64(11)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input = torch.linspace(-5, 5, steps=6, dtype=torch.float32).reshape(2, 3).numpy()
    other = np.int32(-7)
    out = np.empty_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input = torch.tensor([[[1, -1], [2, -2]], [[3, -3], [4, -4]]], dtype=torch.int32).numpy()
    other = np.int64(9)
    out = np.zeros_like(input)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.multiply_3"] = multiply_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.multiply_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.multiply_3'.")


check_valid('torch.multiply', generated_inputs['torch.multiply_3'], lib="torch", suffix=3)
