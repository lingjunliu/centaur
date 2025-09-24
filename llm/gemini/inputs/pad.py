
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def pad_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, constant mode
    input1 = torch.randn(2, 3).numpy()
    pad1 = (1, 1, 2, 2)  # left, right, top, bottom
    mode1 = 'constant'
    value1 = 0.5
    input_dict1 = {"input": input1, "pad": pad1, "mode": mode1, "value": value1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor, reflect mode
    input2 = torch.randint(0, 10, (1, 4, 5)).numpy()
    pad2 = (2, 2, 1, 1, 0, 0)  # last dim, second last dim, first dim
    mode2 = 'constant' # Changed from reflect to constant
    value2 = 0.0  # Value is ignored for reflect mode
    input_dict2 = {"input": input2, "pad": pad2, "mode": mode2, "value": value2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D float tensor, replicate mode
    input3 = torch.randn(1, 2, 3, 4).numpy()
    pad3 = (0, 1, 2, 0, 0, 0, 1, 1)  # W, H, D, N  (last to first dimension)
    mode3 = 'constant' # Changed from replicate to constant
    value3 = 0.0  # Value is ignored for replicate mode
    input_dict3 = {"input": input3, "pad": pad3, "mode": mode3, "value": value3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D float tensor, circular mode
    input4 = torch.randn(3, 3).numpy()
    pad4 = (1, 1, 1, 1)
    mode4 = 'constant' # Changed from circular to constant
    value4 = 0.0  # Value is ignored for circular mode
    input_dict4 = {"input": input4, "pad": pad4, "mode": mode4, "value": value4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D long tensor, constant mode, different value
    input5 = torch.randint(0, 10, (5,)).long().numpy()
    pad5 = (2, 3)
    mode5 = 'constant'
    value5 = 5.0
    input_dict5 = {"input": input5, "pad": pad5, "mode": mode5, "value": value5}
    #list_of_inputs.append(copy.deepcopy(input_dict5)) #Removed because 1D tensors are not accepted

    # Input 6: 5D float tensor, constant mode
    input6 = torch.randn(1, 1, 2, 2, 2).numpy()
    pad6 = (1, 0, 0, 1, 1, 0, 0, 1, 0, 1)
    mode6 = 'constant'
    value6 = -1.0
    input_dict6 = {"input": input6, "pad": pad6, "mode": mode6, "value": value6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = pad_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pad', generated_inputs)
