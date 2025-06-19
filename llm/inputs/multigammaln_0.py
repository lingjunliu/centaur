
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multigammaln_inputs():
    list_of_inputs = []

    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    p = 2

    input_dict = {
        "input": input,
        "p": np.int64(p),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    p = 1

    input_dict = {
        "input": input,
        "p": np.int64(p),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = (torch.arange(1, 5, dtype=torch.float32) * 2).numpy()
    p = 3

    input_dict = {
        "input": input,
        "p": np.int64(p),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = (torch.rand(2, 2, 2) * 5).numpy()
    p = 2
    input_dict = {
        "input": input,
        "p": np.int64(p),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, larger p
    input = (torch.rand(3) * 10 + 5).numpy()
    p = 5
    input_dict = {
        "input": input,
        "p": np.int64(p),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.multigammaln"] = multigammaln_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.multigammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.multigammaln'.")

check_valid('torch.special.multigammaln', generated_inputs['torch.special.multigammaln'], lib="torch", suffix=0)
