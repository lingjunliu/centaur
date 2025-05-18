
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def pixel_shuffle_inputs():
    list_of_inputs = []

    input = torch.randn(1, 9, 4, 4).numpy()
    upscale_factor = 3
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4, 8, 8).numpy()
    upscale_factor = 2
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 16, 2, 2).numpy()
    upscale_factor = 4
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 25, 5, 5).numpy()
    upscale_factor = 5
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 49, 7, 7).numpy()
    upscale_factor = 7
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = pixel_shuffle_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PixelShuffle', list_of_inputs)
