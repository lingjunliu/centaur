
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def broadcast_shapes_inputs():
    list_of_inputs = []

    # Test case 1: Simple broadcast
    shapes = [(2, 3), (2, 1)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Different dimensions
    shapes = [(5, 4, 3), (3,)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Empty tuple
    shapes = [(5, 4, 3), ()]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: More than two shapes
    shapes = [(2, 3, 4), (2, 1, 4), (2, 3, 1)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: No broadcasting needed
    shapes = [(2, 3, 4), (2, 3, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Broadcasting with scalar
    shapes = [(5, 4, 3), (1,)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Multiple dimensions requiring broadcasting
    shapes = [(1, 2, 3, 4), (5, 2, 1, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 8: One shape provided
    shapes = [(5,)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = broadcast_shapes_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('broadcast_shapes', generated_inputs)
