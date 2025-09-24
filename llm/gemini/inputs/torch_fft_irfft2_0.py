
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def irfft2_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(4, 5, dtype=torch.complex64).numpy()
    s1 = (4, 8)
    dim1 = (-2, -1)
    norm1 = "backward"
    out1 = torch.empty(4, 8).numpy()
    input_dict1 = {"input": input1, "s": s1, "dim": dim1, "norm": norm1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    s2 = (2, 4)
    dim2 = (-2, -1)
    norm2 = "forward"
    out2 = torch.empty(2, 4).numpy()
    input_dict2 = {"input": input2, "s": s2, "dim": dim2, "norm": norm2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = torch.randn(5, 6, dtype=torch.complex64).numpy()
    s3 = (5, 10)
    dim3 = (-2, -1)
    norm3 = "ortho"
    out3 = torch.empty(5, 10).numpy()
    input_dict3 = {"input": input3, "s": s3, "dim": dim3, "norm": norm3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4
    input4 = torch.randn(3, 4, dtype=torch.complex64).numpy()
    s4 = (3, 6)
    dim4 = (-2, -1)
    norm4 = "backward"
    out4 = torch.empty(3, 6).numpy()
    input_dict4 = {"input": input4, "s": s4, "dim": dim4, "norm": norm4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    s5 = (2, 2)
    dim5 = (-2, -1)
    norm5 = "forward"
    out5 = torch.empty(2, 2).numpy()
    input_dict5 = {"input": input5, "s": s5, "dim": dim5, "norm": norm5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = torch.randn(6, 7, dtype=torch.complex64).numpy()
    s6 = (6, 12)
    dim6 = (-2, -1)
    norm6 = "ortho"
    out6 = torch.empty(6, 12).numpy()
    input_dict6 = {"input": input6, "s": s6, "dim": dim6, "norm": norm6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7
    input7 = torch.randn(1, 1, dtype=torch.complex64).numpy()
    s7 = (1, 1)
    dim7 = (-2, -1)
    norm7 = "backward"
    out7 = torch.empty(1, 1).numpy()
    input_dict7 = {"input": input7, "s": s7, "dim": dim7, "norm": norm7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = torch.randn(7, 8, dtype=torch.complex64).numpy()
    s8 = (7, -1)
    dim8 = (-2, -1)
    norm8 = "forward"
    out8 = torch.empty(7, 14).numpy()
    input_dict8 = {"input": input8, "s": s8, "dim": dim8, "norm": norm8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = torch.randn(8, 9, dtype=torch.complex64).numpy()
    s9 = (8, 16)
    dim9 = (-2, -1)
    norm9 = "ortho"
    out9 = torch.empty(8, 16).numpy()
    input_dict9 = {"input": input9, "s": s9, "dim": dim9, "norm": norm9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = torch.randn(9, 10, dtype=torch.complex64).numpy()
    s10 = (9, 18)
    dim10 = (-2, -1)
    norm10 = "backward"
    out10 = torch.empty(9, 18).numpy()
    input_dict10 = {"input": input10, "s": s10, "dim": dim10, "norm": norm10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.irfft2"] = irfft2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.irfft2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.irfft2'.")

check_valid('torch.fft.irfft2', generated_inputs['torch.fft.irfft2'], lib="torch", suffix=0)
