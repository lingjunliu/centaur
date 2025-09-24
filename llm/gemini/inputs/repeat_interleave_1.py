
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def repeat_interleave_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, positive repeat
    input_tensor = torch.tensor([1, 2, 3])
    repeats = np.int64(2)
    dim = np.int64(0)
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, positive repeat
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats = np.int64(3)
    dim = np.int64(0)
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, repeat along dim=1
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats = np.int64(2)
    dim = np.int64(1)
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, positive repeat
    input_tensor = torch.randn(2, 3, 4)
    repeats = np.int64(2)
    dim = np.int64(1)
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor with large repeat number
    input_tensor = torch.tensor([1])
    repeats = np.int64(100)
    dim = np.int64(0)
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.repeat_interleave_1"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.repeat_interleave_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_1'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_1'], lib="torch", suffix=1)
