
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gammaln_inputs():
    list_of_inputs = []

    # Input 1: Positive scalar
    input_tensor = torch.tensor(2.5).numpy()
    out_tensor = torch.tensor(0.0).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of positive numbers
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of positive numbers
    input_tensor = torch.tensor([[0.5, 1.5], [2.5, 3.5]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger values
    input_tensor = torch.tensor([10.0, 20.0, 30.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Smaller values close to zero (but still positive)
    input_tensor = torch.tensor([0.1, 0.2, 0.3]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.gammaln"] = gammaln_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.gammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.gammaln'.")

check_valid('torch.special.gammaln', generated_inputs['torch.special.gammaln'], lib="torch", suffix=0)
