
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def sum_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, -2.5, 3.5], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, 2, 3], [-4, 5, -6]], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": False, "dtype": np.dtype('int64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 4).double().numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": True, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 256, (2, 3, 4), dtype=torch.uint8).numpy()
    input_dict = {"input": input_arr, "dim": -1, "keepdim": True, "dtype": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.rand(2, 2, 3) > 0.5).numpy()
    input_dict = {"input": input_arr, "dim": 2, "keepdim": False, "dtype": np.dtype('int64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(-1000, 1000, (2, 2, 2, 3), dtype=torch.int16).numpy()
    input_dict = {"input": input_arr, "dim": -2, "keepdim": True, "dtype": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 5, 6).to(torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": False, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": True, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-100, 200], [300, -400], [0, 0]], dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": False, "dtype": np.dtype('int64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(2 * 3 * 4).view(2, 3, 4).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": True, "dtype": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones(7, dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "dim": 0, "keepdim": True, "dtype": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[float('nan'), 1.0], [2.0, -3.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 1, "keepdim": True, "dtype": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sum_2"] = sum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sum_2'.")


check_valid('torch.sum', generated_inputs['torch.sum_2'], lib="torch", suffix=2)
