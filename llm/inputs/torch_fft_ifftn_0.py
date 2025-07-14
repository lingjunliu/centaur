
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_fft_ifftn_inputs():
    list_of_inputs = []

    # Input 1: Basic complex tensor
    input_tensor = torch.complex(torch.randn(4, 4), torch.randn(4, 4)).numpy()
    input_dict = {"input": input_tensor, "s": None, "dim": None, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specifying s (signal size)
    input_tensor = torch.complex(torch.randn(8, 8), torch.randn(8, 8)).numpy()
    s_val = (4, 4)
    input_dict = {"input": input_tensor, "s": s_val, "dim": None, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specifying dim
    input_tensor = torch.complex(torch.randn(2, 4, 8), torch.randn(2, 4, 8)).numpy()
    dim_val = (1, 2)
    input_dict = {"input": input_tensor, "s": None, "dim": dim_val, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Specifying norm
    input_tensor = torch.complex(torch.randn(4, 4), torch.randn(4, 4)).numpy()
    norm_val = "forward"
    input_dict = {"input": input_tensor, "s": None, "dim": None, "norm": norm_val, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All parameters specified except out
    input_tensor = torch.complex(torch.randn(8, 8), torch.randn(8, 8)).numpy()
    s_val = (4, 4)
    dim_val = (0, 1)
    norm_val = "ortho"
    input_dict = {"input": input_tensor, "s": s_val, "dim": dim_val, "norm": norm_val, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor
    input_tensor = torch.complex(torch.randn(2, 4, 6), torch.randn(2, 4, 6)).numpy()
    input_dict = {"input": input_tensor, "s": None, "dim": None, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: s with -1 and other dimension
    input_tensor = torch.complex(torch.randn(4, 4), torch.randn(4, 4)).numpy()
    s_val = (4, -1)
    input_dict = {"input": input_tensor, "s": s_val, "dim": None, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dims
    input_tensor = torch.complex(torch.randn(2, 3, 4), torch.randn(2, 3, 4)).numpy()
    dim_val = (0, 2)
    input_dict = {"input": input_tensor, "s": None, "dim": dim_val, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: s smaller than input size
    input_tensor = torch.complex(torch.randn(5, 5), torch.randn(5, 5)).numpy()
    s_val = (3, 3)
    input_dict = {"input": input_tensor, "s": s_val, "dim": None, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: No s, no dim
    input_tensor = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).numpy()
    input_dict = {"input": input_tensor, "s": None, "dim": None, "norm": "backward", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ifftn"] = torch_fft_ifftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ifftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifftn'.")

check_valid('torch.fft.ifftn', generated_inputs['torch.fft.ifftn'], lib="torch", suffix=0)
