
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_pow_3_inputs():
    list_of_inputs = []

    # Case 1: Basic case with positive exponent
    self = 2.0
    exponent = torch.arange(1., 5.).numpy()
    input_dict = {"self": self, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Negative exponent
    self = 3.0
    exponent = torch.tensor([-1.0, 0.0, 1.0, 2.0]).numpy()
    input_dict = {"self": self, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Large exponent values
    self = 1.5
    exponent = torch.tensor([10.0, 20.0, 30.0]).numpy()
    input_dict = {"self": self, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Exponent with zeros
    self = 4.0
    exponent = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"self": self, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different data type for exponent (int)
    self = 2.0
    exponent = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    input_dict = {"self": self, "exponent": exponent, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.pow_3"] = torch_pow_3_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.pow_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_3'.")

check_valid('torch.pow', generated_inputs['torch.pow_3'], lib="torch")
