
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def pixel_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic test case with float32 and upscale_factor = 2
    input1 = torch.randn(1, 4, 5, 5).numpy()
    upscale_factor1 = 2
    input_dict1 = {"input": input1, "upscale_factor": upscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different input size, upscale_factor = 3
    input2 = torch.randn(1, 9, 4, 4).numpy()
    upscale_factor2 = 3
    input_dict2 = {"input": input2, "upscale_factor": upscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Larger input with upscale_factor = 4
    input3 = torch.randn(2, 16, 8, 8).numpy()
    upscale_factor3 = 4
    input_dict3 = {"input": input3, "upscale_factor": upscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Single channel input
    input4 = torch.randn(1, 4, 5, 5).numpy()
    upscale_factor4 = 2
    input_dict4 = {"input": input4, "upscale_factor": upscale_factor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different batch size, float64 input
    input5 = torch.randn(3, 4, 6, 6, dtype=torch.float64).numpy()
    upscale_factor5 = 2
    input_dict5 = {"input": input5, "upscale_factor": upscale_factor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Channel divisible by upscale_factor squared
    input6 = torch.randn(1, 8, 5, 5).numpy()
    upscale_factor6 = 2
    input_dict6 = {"input": input6, "upscale_factor": upscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Input with height and width of size 1
    input7 = torch.randn(1, 4, 1, 1).numpy()
    upscale_factor7 = 2
    input_dict7 = {"input": input7, "upscale_factor": upscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Different channel size and upscale factor
    input8 = torch.randn(1, 36, 4, 4).numpy()
    upscale_factor8 = 6
    input_dict8 = {"input": input8, "upscale_factor": upscale_factor8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Another valid case
    input9 = torch.randn(2, 16, 5, 5).numpy()
    upscale_factor9 = 4
    input_dict9 = {"input": input9, "upscale_factor": upscale_factor9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = pixel_shuffle_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pixel_shuffle', generated_inputs)
