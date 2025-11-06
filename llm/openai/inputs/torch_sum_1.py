
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sum_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, -2.5, 3.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones((2, 3), dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(24, dtype=torch.int16).view(2, 3, 4).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 4, 4, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[True, False], [True, True]], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 256, (5,), dtype=torch.uint8).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.randn(2, 3, dtype=torch.float32)
    imag = torch.randn(2, 3, dtype=torch.float32)
    input_arr = torch.complex(real, imag).numpy()
    input_dict = {"input": input_arr, "dtype": torch.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor(42, dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([-128, -1, 0, 127], dtype=torch.int8).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sum_1"] = sum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sum_1'.")


check_valid('torch.sum', generated_inputs['torch.sum_1'], lib="torch", suffix=1)
