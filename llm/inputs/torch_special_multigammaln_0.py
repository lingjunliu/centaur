
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def multigammaln_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive values and p = 1
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    p1 = 1
    input_dict1 = {"input": input1, "p": p1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic case with positive values and p = 2
    input2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p2 = 2
    input_dict2 = {"input": input2, "p": p2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional input
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    p3 = 1
    input_dict3 = {"input": input3, "p": p3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger values
    input4 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    p4 = 3
    input_dict4 = {"input": input4, "p": p4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Float16 input
    input5 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    p5 = 1
    input_dict5 = {"input": input5, "p": p5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: different p value
    input6 = np.array([2.5, 3.5, 4.5], dtype=np.float64)
    p6 = 4
    input_dict6 = {"input": input6, "p": p6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Multi-dimensional input with different shape
    input7 = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32)
    p7 = 2
    input_dict7 = {"input": input7, "p": p7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Test single element array
    input8 = np.array([5.0], dtype=np.float64)
    p8 = 1
    input_dict8 = {"input": input8, "p": p8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Test larger p value than input dimension
    input9 = np.array([6.0, 7.0], dtype=np.float32)
    p9 = 1
    input_dict9 = {"input": input9, "p": p9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Larger values with float16
    input10 = np.array([5.0, 10.0, 15.0], dtype=np.float16)
    p10 = 2
    input_dict10 = {"input": input10, "p": p10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
