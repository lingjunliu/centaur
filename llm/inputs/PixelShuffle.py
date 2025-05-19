
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def pixelshuffle_inputs():
    list_of_inputs = []

    input1 = np.random.rand(1, 9, 4, 4).astype(np.float32)
    upscale_factor1 = 3
    input_dict1 = {"input": input1, "upscale_factor": upscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 4, 5, 6).astype(np.float64)
    upscale_factor2 = 2
    input_dict2 = {"input": input2, "upscale_factor": upscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 16, 3, 3).astype(np.int32)
    upscale_factor3 = 4
    input_dict3 = {"input": input3, "upscale_factor": upscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(4, 1, 8, 8).astype(np.float16)
    upscale_factor4 = 1
    input_dict4 = {"input": input4, "upscale_factor": upscale_factor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(1, 25, 2, 2).astype(np.float32)
    upscale_factor5 = 5
    input_dict5 = {"input": input5, "upscale_factor": upscale_factor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(1, 64, 1, 1).astype(np.float32)
    upscale_factor6 = 8
    input_dict6 = {"input": input6, "upscale_factor": upscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(2, 4, 10, 12).astype(np.float32)
    upscale_factor7 = 2
    input_dict7 = {"input": input7, "upscale_factor": upscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = pixelshuffle_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PixelShuffle', generated_inputs)
