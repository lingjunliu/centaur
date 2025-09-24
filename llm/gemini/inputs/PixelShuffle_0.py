
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def pixel_shuffle_inputs():
    list_of_inputs = []

    # Example 1
    upscale_factor = 2
    input_tensor = torch.randn(1, 4, 4, 4).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2
    upscale_factor = 3
    input_tensor = torch.randn(2, 9, 2, 2).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3
    upscale_factor = 4
    input_tensor = torch.randn(1, 16, 3, 3).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4
    upscale_factor = 2
    input_tensor = torch.randn(3, 4, 5, 5).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5
    upscale_factor = 3
    input_tensor = torch.randn(2, 9, 1, 1).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Different sized input
    upscale_factor = 2
    input_tensor = torch.randn(1, 4, 6, 8).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Different batch size
    upscale_factor = 3
    input_tensor = torch.randn(4, 9, 2, 2).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.PixelShuffle"] = pixel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.PixelShuffle', generated_inputs['torch.nn.PixelShuffle'], lib="torch")
