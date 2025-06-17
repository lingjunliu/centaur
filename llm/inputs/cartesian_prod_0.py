
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cartesian_prod_inputs():
    list_of_inputs = []

    tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cartesian_prod"] = cartesian_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cartesian_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cartesian_prod'.")

check_valid('torch.cartesian_prod', generated_inputs['torch.cartesian_prod'], lib="torch")
