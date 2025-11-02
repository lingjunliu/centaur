
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def nansum_inputs():
    list_of_inputs = []

    input = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[1.0, -2.5, np.nan],
                      [3.0, np.nan, 6.5]], dtype=np.float32)
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[[1.0, np.nan], [-np.inf, 2.0]],
                      [[np.nan, 3.0], [4.0, 5.0]]], dtype=np.float16)
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array(np.nan, dtype=np.float32)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.empty((2, 0, 3), dtype=np.float32)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([-5, 0, 7, -3], dtype=np.int32)
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[255, 0, 1],
                      [2, 3, 4]], dtype=np.uint8)
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([[[[1.0, np.nan, -1.0],
                        [2.0, 3.0, np.nan]]],
                      [[[np.nan, 0.0, 5.0],
                        [6.0, -7.5, 8.0]]]], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([1e20, 1e20, np.nan, -1e20], dtype=np.float32)
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4).swapaxes(1, 2)
    arr[0, 1, 2] = np.nan
    input = arr
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_1'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_1'], lib="torch", suffix=1)
