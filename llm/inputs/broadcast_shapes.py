
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def broadcast_shapes_inputs():
    list_of_inputs = []

    shapes = [(2, 3), (1, 3)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shapes = [(2, 1, 4), (1, 3, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shapes = [(5, 4), (1, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shapes = [(15, 3, 5), (1, 3, 5)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shapes = [(8, 1, 6, 1), (8, 7, 6, 5)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = broadcast_shapes_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('broadcast_shapes', list_of_inputs)
