
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def flip_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dims = (2, 3)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5, 5).numpy()
    dims = (1,)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.arange(24).reshape(2, 3, 4).numpy()
    dims = (0, 2)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = np.array([[1, 2], [3, 4]]).astype(np.float32)
    dims = (0, 1)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 2).numpy()
    dims = (0,)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = flip_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('flip', list_of_inputs)
