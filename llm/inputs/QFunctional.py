
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def QFunctional_inputs():
    list_of_inputs = []

    scale1 = 0.5
    zero_point1 = 10
    dtype1 = torch.quint8
    x1 = torch.randn(2, 3, 4, 5)
    qx1 = torch.quantize_per_tensor(x1, scale1, zero_point1, dtype1)

    scale2 = 0.75
    zero_point2 = 5
    dtype2 = torch.quint8
    x2 = torch.randn(2, 3, 4, 5)
    qx2 = torch.quantize_per_tensor(x2, scale2, zero_point2, dtype2)
    
    input_dict = {
        "x": qx1,
        "y": qx2,
        "scale": torch.tensor(0.6),
        "zero_point": torch.tensor(7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    scale1 = 0.3
    zero_point1 = 12
    dtype1 = torch.qint8
    x1 = torch.randn(3, 5, 7)
    qx1 = torch.quantize_per_tensor(x1, scale1, zero_point1, dtype1)

    scale2 = 0.6
    zero_point2 = -3
    dtype2 = torch.qint8
    x2 = torch.randn(3, 5, 7)
    qx2 = torch.quantize_per_tensor(x2, scale2, zero_point2, dtype2)
    
    input_dict = {
        "x": qx1,
        "y": qx2,
        "scale": torch.tensor(0.4),
        "zero_point": torch.tensor(-5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = QFunctional_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('QFunctional', generated_inputs)
