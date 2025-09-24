
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def topk_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    k = 3
    dim = 0
    largest = True
    sorted = True
    out = (np.array([1.0], dtype=np.float32), np.array([1]))

    input_dict = {
        "input": input,
        "k": k,
        "dim": dim,
        "largest": largest,
        "sorted": sorted,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    k = 2
    dim = 1
    largest = False
    sorted = False
    out = (np.array([1.0], dtype=np.float32), np.array([1]))

    input_dict = {
        "input": input,
        "k": k,
        "dim": dim,
        "largest": largest,
        "sorted": sorted,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(3, 4, 5).numpy()
    k = 1
    dim = 2
    largest = True
    sorted = True
    out = (np.array([1.0], dtype=np.float32), np.array([1]))

    input_dict = {
        "input": input,
        "k": k,
        "dim": dim,
        "largest": largest,
        "sorted": sorted,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0]).numpy()
    k = 4
    dim = 0
    largest = False
    sorted = True
    out = (np.array([1.0], dtype=np.float32), np.array([1]))

    input_dict = {
        "input": input,
        "k": k,
        "dim": dim,
        "largest": largest,
        "sorted": sorted,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.randn(2, 2).numpy()
    k = 2
    dim = 1
    largest = True
    sorted = False
    out = (np.array([1.0], dtype=np.float32), np.array([1]))

    input_dict = {
        "input": input,
        "k": k,
        "dim": dim,
        "largest": largest,
        "sorted": sorted,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.topk"] = topk_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.topk' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.topk'.")

check_valid('torch.topk', generated_inputs['torch.topk'], lib="torch", suffix=0)
