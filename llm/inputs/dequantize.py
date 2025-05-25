
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def dequantize_inputs():
    list_of_inputs = []

    scale = 0.5
    zero_point = 10
    qint8_tensor = torch.quantize_per_tensor(torch.randn(3, 4).float(), scale, zero_point, torch.qint8)
    input_dict = {"tensor": qint8_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    scale = 0.2
    zero_point = 5
    quint8_tensor = torch.quantize_per_tensor(torch.rand(2, 2).float(), scale, zero_point, torch.quint8)
    input_dict = {"tensor": quint8_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    scale = 0.1
    zero_point = 0
    qint32_tensor = torch.quantize_per_tensor(torch.randn(1, 5).float(), scale, zero_point, torch.qint32)
    input_dict = {"tensor": qint32_tensor}
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
