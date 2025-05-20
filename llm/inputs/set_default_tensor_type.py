
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def set_default_tensor_type_inputs():
    list_of_inputs = []

    input1 = {"t": 'torch.FloatTensor'}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {"t": 'torch.DoubleTensor'}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {"t": 'torch.HalfTensor'}
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {"t": 'torch.ByteTensor'}
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {"t": 'torch.ShortTensor'}
    list_of_inputs.append(copy.deepcopy(input5))

    return list_of_inputs

generated_inputs = set_default_tensor_type_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_default_tensor_type', generated_inputs)
