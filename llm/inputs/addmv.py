
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addmv_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, beta and alpha
    input = torch.randn(5).numpy()
    mat = torch.randn(5, 4).numpy()
    vec = torch.randn(4).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 0.5, 'alpha': 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    input = torch.randn(4).numpy() * -1
    mat = torch.randn(4, 3).numpy() * -1
    vec = torch.randn(3).numpy() * -1
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Zero values
    input = torch.zeros(2).numpy()
    mat = torch.zeros(2, 5).numpy()
    vec = torch.zeros(5).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Double type
    input = torch.randn(6, dtype=torch.float64).numpy()
    mat = torch.randn(6, 2, dtype=torch.float64).numpy()
    vec = torch.randn(2, dtype=torch.float64).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Beta and alpha are zeros.
    input = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 0.0, 'alpha': 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = addmv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addmv', generated_inputs)
