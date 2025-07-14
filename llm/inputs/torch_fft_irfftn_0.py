
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def irfftn_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input1 = torch.randn(5, 5, 3).numpy()
    s1 = (5, 5, 5)
    dim1 = (0, 1, 2)
    norm1 = "backward"
    out1 = torch.randn(5, 5, 5).numpy()

    input_dict1 = {
        "input": input1,
        "s": s1,
        "dim": dim1,
        "norm": norm1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: No output tensor
    input2 = torch.randn(4, 4, 3).numpy()
    s2 = (4, 4, 5)
    dim2 = (0, 1, 2)
    norm2 = "forward"
    out2 = torch.randn(4, 4, 5).numpy()
    input_dict2 = {
        "input": input2,
        "s": s2,
        "dim": dim2,
        "norm": norm2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different norm
    input3 = torch.randn(3, 3, 2).numpy()
    s3 = (3, 3, 3)
    dim3 = (0, 1, 2)
    norm3 = "ortho"
    out3 = torch.randn(3, 3, 3).numpy()
    input_dict3 = {
        "input": input3,
        "s": s3,
        "dim": dim3,
        "norm": norm3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: s=None
    input4 = torch.randn(2, 2, 2).numpy()
    s4 = None
    dim4 = None
    norm4 = "backward"
    out4 = torch.randn(2,2,2).numpy()
    input_dict4 = {
        "input": input4,
        "s": s4,
        "dim": dim4,
        "norm": norm4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: dim=None
    input5 = torch.randn(6, 6, 4).numpy()
    s5 = (6, 6, 7)
    dim5 = None
    norm5 = "backward"
    out5 = torch.randn(6, 6, 7).numpy()
    input_dict5 = {
        "input": input5,
        "s": s5,
        "dim": dim5,
        "norm": norm5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: s and dim = None
    input6 = torch.randn(7, 7, 5).numpy()
    s6 = None
    dim6 = None
    norm6 = "backward"
    out6 = torch.randn(7,7,8).numpy()
    input_dict6 = {
        "input": input6,
        "s": s6,
        "dim": dim6,
        "norm": norm6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 1D input
    input7 = torch.randn(4).numpy()
    s7 = (7,)
    dim7 = (0,)
    norm7 = "backward"
    out7 = torch.randn(7,).numpy()
    input_dict7 = {
        "input": input7,
        "s": s7,
        "dim": dim7,
        "norm": norm7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Negative values in input
    input8 = torch.randn(2, 2, 2)
    input8[0,0,0] = -1
    input8 = input8.numpy()
    s8 = (2, 2, 3)
    dim8 = (0, 1, 2)
    norm8 = "backward"
    out8 = torch.randn(2, 2, 3).numpy()
    input_dict8 = {
        "input": input8,
        "s": s8,
        "dim": dim8,
        "norm": norm8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: s with -1
    input9 = torch.randn(3, 4, 3).numpy()
    s9 = (3, -1, 5)
    dim9 = (0, 1, 2)
    norm9 = "backward"
    out9 = torch.randn(3, 4, 5).numpy()
    input_dict9 = {
        "input": input9,
        "s": s9,
        "dim": dim9,
        "norm": norm9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Different dims
    input10 = torch.randn(2, 4).numpy()
    s10 = (2, 7)
    dim10 = (0, 1)
    norm10 = "backward"
    out10 = torch.randn(2, 7).numpy()
    input_dict10 = {
        "input": input10,
        "s": s10,
        "dim": dim10,
        "norm": norm10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: empty dim
    input11 = torch.randn(2, 4).numpy()
    s11 = (2, 7)
    dim11 = ()
    norm11 = "backward"
    out11 = torch.randn(2, 7).numpy()
    input_dict11 = {
        "input": input11,
        "s": s11,
        "dim": dim11,
        "norm": norm11,
        "out": out11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: empty s
    input12 = torch.randn(2, 4).numpy()
    s12 = ()
    dim12 = ()
    norm12 = "backward"
    out12 = torch.randn(2, 4).numpy()
    input_dict12 = {
        "input": input12,
        "s": s12,
        "dim": dim12,
        "norm": norm12,
        "out": out12
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))
    
    # Input 13: complex input
    input13 = (torch.randn(2, 4) + 1j * torch.randn(2, 4)).numpy()
    s13 = (2, 7)
    dim13 = (0, 1)
    norm13 = "backward"
    out13 = torch.randn(2, 7).numpy()
    input_dict13 = {
        "input": input13,
        "s": s13,
        "dim": dim13,
        "norm": norm13,
        "out": out13
    }
    list_of_inputs.append(copy.deepcopy(input_dict13))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.irfftn"] = irfftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.irfftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.irfftn'.")

check_valid('torch.fft.irfftn', generated_inputs['torch.fft.irfftn'], lib="torch", suffix=0)
