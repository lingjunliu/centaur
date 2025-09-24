
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def pixel_unshuffle_inputs():
    list_of_inputs = []

    downscale_factor = 2
    input1 = torch.randn(1, 1, 4, 4).numpy()
    input_dict1 = {
        "downscale_factor": downscale_factor,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    downscale_factor = 3
    input2 = torch.randn(1, 1, 12, 12).numpy()
    input_dict2 = {
        "downscale_factor": downscale_factor,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    downscale_factor = 4
    input3 = torch.randn(2, 3, 20, 20).numpy()
    input_dict3 = {
        "downscale_factor": downscale_factor,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    downscale_factor = 2
    input4 = torch.randn(1, 5, 6, 6).numpy()
    input_dict4 = {
        "downscale_factor": downscale_factor,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    downscale_factor = 3
    input5 = torch.randn(4, 2, 9, 9).numpy()
    input_dict5 = {
        "downscale_factor": downscale_factor,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    downscale_factor = 2
    input6 = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict6 = {
        "downscale_factor": downscale_factor,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    downscale_factor = 2
    input7 = torch.randn(1, 1, 8, 8).numpy()
    input_dict7 = {
        "downscale_factor": downscale_factor,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = pixel_unshuffle_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PixelUnshuffle', generated_inputs)
