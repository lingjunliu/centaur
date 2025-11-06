
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_sum_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1, -2, 3, -4, 5], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dim": (0,), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": (1,), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(24, dtype=torch.int64).view(2, 3, 4).numpy()
    input_dict = {"input": input_arr, "dim": (1, 2), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 2, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": (0, -1), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[True, False, True], [False, True, True]], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "dim": (0,), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 4, 4, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": (-2,), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(12, dtype=torch.uint8).view(3, 2, 2).numpy()
    input_dict = {"input": input_arr, "dim": (0, 2), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones((2, 2, 2, 2, 2), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": (-5, -3, -1), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-5]], dtype=torch.int8).repeat(1, 1).numpy()
    input_dict = {"input": input_arr, "dim": (0, 1), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(3, 3, dtype=torch.float32) + 1j * torch.randn(3, 3, dtype=torch.float32)).numpy()
    input_dict = {"input": input_arr, "dim": (0,), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty((0, 3, 2), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": (0, 2), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(12, dtype=torch.float64).view(3, 4)[:, ::2].numpy()
    input_dict = {"input": input_arr, "dim": (1,), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sum_3"] = torch_sum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sum_3'.")


check_valid('torch.sum', generated_inputs['torch.sum_3'], lib="torch", suffix=3)
