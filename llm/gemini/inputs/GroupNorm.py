
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def groupnorm_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 6, 5, 5).numpy()
    num_groups1 = 3
    num_channels1 = 6
    eps1 = 1e-5
    affine1 = True

    input_dict1 = {
        "input": input1,
        "num_groups": num_groups1,
        "num_channels": num_channels1,
        "eps": eps1,
        "affine": affine1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 8, 4, 4).numpy()
    num_groups2 = 4
    num_channels2 = 8
    eps2 = 1e-8
    affine2 = False

    input_dict2 = {
        "input": input2,
        "num_groups": num_groups2,
        "num_channels": num_channels2,
        "eps": eps2,
        "affine": affine2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 12, 3, 3).numpy()
    num_groups3 = 1
    num_channels3 = 12
    eps3 = 1e-3
    affine3 = True

    input_dict3 = {
        "input": input3,
        "num_groups": num_groups3,
        "num_channels": num_channels3,
        "eps": eps3,
        "affine": affine3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 16, 2, 2).numpy()
    num_groups4 = 8
    num_channels4 = 16
    eps4 = 1e-6
    affine4 = False

    input_dict4 = {
        "input": input4,
        "num_groups": num_groups4,
        "num_channels": num_channels4,
        "eps": eps4,
        "affine": affine4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 4, 1, 1, 1).numpy()
    num_groups5 = 2
    num_channels5 = 4
    eps5 = 1e-4
    affine5 = True

    input_dict5 = {
        "input": input5,
        "num_groups": num_groups5,
        "num_channels": num_channels5,
        "eps": eps5,
        "affine": affine5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 32, 10).numpy()
    num_groups6 = 16
    num_channels6 = 32
    eps6 = 1e-2
    affine6 = False
    input_dict6 = {
        "input": input6,
        "num_groups": num_groups6,
        "num_channels": num_channels6,
        "eps": eps6,
        "affine": affine6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(1, 2, 3).numpy()
    num_groups7 = 1
    num_channels7 = 2
    eps7 = 1e-7
    affine7 = True
    input_dict7 = {
        "input": input7,
        "num_groups": num_groups7,
        "num_channels": num_channels7,
        "eps": eps7,
        "affine": affine7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = groupnorm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('GroupNorm', generated_inputs)
