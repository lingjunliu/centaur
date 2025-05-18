
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def complex_inputs():
    list_of_inputs = []

    real = torch.randn(3, 4).numpy()
    imag = torch.randn(3, 4).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.randn(2, 2, 2).numpy()
    imag = torch.randn(2, 2, 2).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.array([1.0, 2.0, 3.0])
    imag = np.array([4.0, 5.0, 6.0])
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([[1.0, 2.0], [3.0, 4.0]])
    imag = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.zeros(5).numpy()
    imag = torch.ones(5).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = complex_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('complex', list_of_inputs)
