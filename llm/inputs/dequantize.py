
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def dequantize_inputs():
    list_of_inputs = []

    qint8_tensor = torch.quantize_per_tensor(torch.randn(3, 4), 0.5, 10, torch.quint8)
    input_dict = {"tensor": qint8_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    quint8_tensor = torch.quantize_per_tensor(torch.rand(2, 2), 0.2, 5, torch.quint8)
    input_dict = {"tensor": quint8_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = dequantize_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('dequantize', generated_inputs)
