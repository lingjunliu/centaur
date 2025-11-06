
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_unshuffle_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(3, 8, 10).numpy()
    input_dict = {"downscale_factor": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.random.randint(-50, 51, size=(1, 12, 15), dtype=np.int32)
    input_dict = {"downscale_factor": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 1, 12, 12, dtype=torch.float64).numpy()
    input_dict = {"downscale_factor": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(2, 1, 3, 16, 8, dtype=torch.float16).numpy()
    input_dict = {"downscale_factor": 4, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = np.arange(2 * 5 * 6 * 8, dtype=np.int8).reshape(2, 5, 6, 8) - 64
    input_dict = {"downscale_factor": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.randn(1, 7, 10, 15).numpy()
    input_dict = {"downscale_factor": 5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.linspace(-1.0, 1.0, num=3 * 2 * 7 * 9, dtype=np.float32).reshape(3, 2, 7, 9)
    input_dict = {"downscale_factor": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = np.random.randint(-1000, 1000, size=(3, 2, 1, 12, 18), dtype=np.int16)
    input_dict = {"downscale_factor": 6, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.random.randn(4, 2, 14).astype(np.float32)
    input_arr[0, 0, 0] = np.nan
    input_arr[1, 1, 1] = np.inf
    input_arr[2, 0, 2] = -np.inf
    input_dict = {"downscale_factor": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = (np.ones((1, 3, 14, 21), dtype=np.float64) * 5.0)
    input_dict = {"downscale_factor": 7, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (non-contiguous view, valid H and W)
    base = np.random.randn(2, 5, 6, 8).astype(np.float32)
    input_arr = base.transpose(0, 1, 3, 2)  # shape (2,5,8,6): H=8, W=6
    input_dict = {"downscale_factor": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(2, 7, 9, 6).numpy()
    input_dict = {"downscale_factor": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.PixelUnshuffle"] = pixel_unshuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.PixelUnshuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PixelUnshuffle'.")


check_valid('torch.nn.PixelUnshuffle', generated_inputs['torch.nn.PixelUnshuffle'], lib="torch", suffix=0)
