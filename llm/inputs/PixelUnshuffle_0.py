
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def pixel_unshuffle_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with downscale_factor = 2
    input1 = torch.randn(1, 1, 4, 4).numpy()
    downscale_factor1 = 2
    input_dict1 = {"downscale_factor": downscale_factor1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Different channel size
    input2 = torch.randn(1, 3, 6, 6).numpy()
    downscale_factor2 = 3
    input_dict2 = {"downscale_factor": downscale_factor2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Batch size > 1
    input3 = torch.randn(2, 1, 8, 8).numpy()
    downscale_factor3 = 4
    input_dict3 = {"downscale_factor": downscale_factor3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Larger input size
    input4 = torch.randn(1, 2, 10, 10).numpy()
    downscale_factor4 = 5
    input_dict4 = {"downscale_factor": downscale_factor4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Float64 input
    input5 = torch.randn(1, 1, 4, 4, dtype=torch.float64).numpy()
    downscale_factor5 = 2
    input_dict5 = {"downscale_factor": downscale_factor5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Downscale factor = 1
    input6 = torch.randn(1, 1, 4, 4).numpy()
    downscale_factor6 = 1
    input_dict6 = {"downscale_factor": downscale_factor6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Multi-dimensional input
    input7 = torch.randn(2, 3, 4, 4).numpy()
    downscale_factor7 = 2
    input_dict7 = {"downscale_factor": downscale_factor7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.PixelUnshuffle"] = pixel_unshuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.PixelUnshuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PixelUnshuffle'.")

check_valid('torch.nn.PixelUnshuffle', generated_inputs['torch.nn.PixelUnshuffle'], lib="torch")
