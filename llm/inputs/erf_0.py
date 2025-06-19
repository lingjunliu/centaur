
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def erf_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor
    input = np.array(0.5)
    out = np.array(0.0)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor with negative values
    input = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    out = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = np.array([[0.0, 0.0], [0.0, 0.0]])
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor with different values
    input = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    out = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.float32)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with large values
    input = np.array([10.0, 20.0, 30.0])
    out = np.array([0.0, 0.0, 0.0])
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Tensor with small values
    input = np.array([0.01, 0.02, 0.03])
    out = np.array([0.0, 0.0, 0.0])
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with a mix of positive and negative values
    input = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
    out = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.erf"] = erf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.erf'.")

check_valid('torch.erf', generated_inputs['torch.erf'], lib="torch", suffix=0)
