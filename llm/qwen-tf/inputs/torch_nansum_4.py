
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def nansum_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, 2.0, float('nan'), -3.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1.0, 2.0], [3.0, float('nan')]], dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": True, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[float('nan'), -5.5, 2.0], [1.0, float('nan'), 3.5]], dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": [1], "keepdim": False, "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[1.0, float('nan')], [2.0, 3.0]], [[float('nan'), 4.0], [5.0, float('nan')]]], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": [0, 2], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(24., dtype=torch.float64).reshape(2, 3, 4).numpy()
    input_dict = {"input": input_arr, "dim": [-1], "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    input_arr[0, 1, 2, 3] = np.nan
    input_arr[1, 2, 0, 4] = np.nan
    input_dict = {"input": input_arr, "dim": [1, 3], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-1, 2, -3], [4, -5, 6]], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": False, "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1.0, float('inf')], [float('nan'), -float('inf')]], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": [0, 1], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[-1.5, 2.5], [float('nan'), -3.0]], [[4.0, -2.0], [1.0, float('nan')]]], dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": [1], "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    input_arr[0, 0, 1, 0, 2] = np.nan
    input_arr[1, 0, 2, 0, 3] = np.nan
    input_dict = {"input": input_arr, "dim": [0, 2, 4], "keepdim": True, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(12., dtype=torch.float32).reshape(3, 4).t().numpy()
    input_dict = {"input": input_arr, "dim": [1], "keepdim": False, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-10, 20], [30, -40]], dtype=torch.int16).numpy()
    input_dict = {"input": input_arr, "dim": [0], "keepdim": True, "dtype": torch.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nansum_4"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_4'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_4'], lib="torch", suffix=4)
