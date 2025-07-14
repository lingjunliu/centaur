
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rfftn_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(10, 10).numpy()
    s = None
    dim = None
    norm = None
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(5, 5, 5).numpy()
    s = (8, 8, 8)
    dim = (0, 1, 2)
    norm = "forward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(10, 20).numpy()
    s = (5, 10)
    dim = (0, 1)
    norm = "backward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(3, 4, 5).numpy()
    s = (3, 4, -1)
    dim = (0, 1, 2)
    norm = "ortho"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.randn(2, 2).numpy()
    s = (4, 4)
    dim = (0, 1)
    norm = None
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(7, 8, 9).numpy()
    s = None
    dim = (0, 1)
    norm = "forward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(4, 4).numpy()
    s = (8, 8)
    dim = (0, 1)
    norm = "backward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_tensor = torch.randn(16, 16).numpy()
    s = None
    dim = (0, 1)
    norm = "ortho"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.randn(2, 3, 4).numpy()
    s = None
    dim = (0, 1, 2)
    norm = "forward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = torch.randn(8, 8).numpy()
    s = None
    dim = (0, 1)
    norm = "backward"
    out = None
    input_dict = {"input": input_tensor, "s": s, "dim": dim, "norm": norm, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.rfftn"] = rfftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.rfftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfftn'.")

check_valid('torch.fft.rfftn', generated_inputs['torch.fft.rfftn'], lib="torch", suffix=0)
