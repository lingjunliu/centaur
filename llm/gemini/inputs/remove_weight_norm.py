
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import copy

def remove_weight_norm_inputs():
    list_of_inputs = []

    class MyModule1(nn.Module):
        def __init__(self):
            super(MyModule1, self).__init__()
            self.linear = nn.Linear(10, 20)
            nn_utils.weight_norm(self.linear)
    module1 = MyModule1()
    input_dict1 = {"module": module1, "name": "linear"}
    list_of_inputs.append(input_dict1)
    
    return list_of_inputs

generated_inputs = remove_weight_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('remove_weight_norm', generated_inputs)
