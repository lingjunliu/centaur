
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def zeros_like_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.5, -2.3, 0.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float32, "requires_grad": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, -2, 3], [4, -5, 6]], dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int64, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float64, "requires_grad": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor(5.5, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float16, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float32, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "dtype": torch.bool, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1+2j, -3-4j], [0+0j, 5-6j]], dtype=torch.complex64).numpy()
    input_dict = {"input": input_arr, "dtype": torch.complex64, "requires_grad": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([1+0.5j, -2-1j, 3+4j], dtype=torch.complex128).numpy()
    input_dict = {"input": input_arr, "dtype": torch.complex128, "requires_grad": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 256, (3, 3), dtype=torch.uint8).numpy()
    input_dict = {"input": input_arr, "dtype": torch.uint8, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([-1, 0, 1, -128, 127], dtype=torch.int8).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int8, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 2, 2, 3, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float32, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    sliced = base[:, ::2].numpy()
    input_dict = {"input": sliced, "dtype": torch.float32, "requires_grad": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([100000, -200000, 300000], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dtype": torch.int32, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 0, 5, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dtype": torch.float64, "requires_grad": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.zeros_like"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.zeros_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.zeros_like'.")


check_valid('torch.zeros_like', generated_inputs['torch.zeros_like'], lib="torch", suffix=0)
