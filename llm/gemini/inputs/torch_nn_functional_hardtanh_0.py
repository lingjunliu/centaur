
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hardtanh_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    min_val1 = -1.0
    max_val1 = 1.0
    inplace1 = False
    input_dict1 = {"input": input1, "min_val": min_val1, "max_val": max_val1, "inplace": inplace1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic 2D tensor
    input2 = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]])
    min_val2 = -1.0
    max_val2 = 1.0
    inplace2 = True
    input_dict2 = {"input": input2, "min_val": min_val2, "max_val": max_val2, "inplace": inplace2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different min/max values
    input3 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    min_val3 = -2.0
    max_val3 = 2.0
    inplace3 = False
    input_dict3 = {"input": input3, "min_val": min_val3, "max_val": max_val3, "inplace": inplace3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Min > Max (should still work, clamping) - REMOVED as this causes error
    # input4 = np.array([-1.0, 0.0, 1.0])
    # min_val4 = 1.0
    # max_val4 = -1.0
    # inplace4 = False
    # input_dict4 = {"input": input4, "min_val": min_val4, "max_val": max_val4, "inplace": inplace4}
    # list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 3D tensor
    input5 = np.random.rand(2, 3, 4).astype(np.float32) * 4 - 2
    min_val5 = -0.5
    max_val5 = 1.5
    inplace5 = False
    input_dict5 = {"input": input5, "min_val": min_val5, "max_val": max_val5, "inplace": inplace5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: All positive values
    input6 = np.array([0.5, 1.5, 2.5, 3.5])
    min_val6 = 1.0
    max_val6 = 3.0
    inplace6 = False
    input_dict6 = {"input": input6, "min_val": min_val6, "max_val": max_val6, "inplace": inplace6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: All negative values
    input7 = np.array([-3.5, -2.5, -1.5, -0.5])
    min_val7 = -3.0
    max_val7 = -1.0
    inplace7 = False
    input_dict7 = {"input": input7, "min_val": min_val7, "max_val": max_val7, "inplace": inplace7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Zero min and max
    input8 = np.array([-1.0, 0.0, 1.0])
    min_val8 = 0.0
    max_val8 = 0.0
    inplace8 = False
    input_dict8 = {"input": input8, "min_val": min_val8, "max_val": max_val8, "inplace": inplace8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Larger tensor with different values for min/max
    input9 = np.random.randn(5, 5).astype(np.float32)
    min_val9 = -0.75
    max_val9 = 0.25
    inplace9 = False
    input_dict9 = {"input": input9, "min_val": min_val9, "max_val": max_val9, "inplace": inplace9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: float64 input
    input10 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    min_val10 = -1.0
    max_val10 = 1.0
    inplace10 = False
    input_dict10 = {"input": input10, "min_val": min_val10, "max_val": max_val10, "inplace": inplace10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Small values
    input11 = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])
    min_val11 = -0.001
    max_val11 = 0.001
    inplace11 = False
    input_dict11 = {"input": input11, "min_val": min_val11, "max_val": max_val11, "inplace": inplace11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.hardtanh"] = hardtanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.hardtanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardtanh'.")

check_valid('torch.nn.functional.hardtanh', generated_inputs['torch.nn.functional.hardtanh'], lib="torch", suffix=0)
