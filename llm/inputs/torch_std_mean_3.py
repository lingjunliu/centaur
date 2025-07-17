
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def std_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dim1 = 0
    unbiased1 = True
    keepdim1 = False
    out1 = (np.array([]), np.array([]))
    input_dict1 = {"input": input1, "dim": dim1, "unbiased": unbiased1, "keepdim": keepdim1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, unbiased = False
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    dim2 = 0
    unbiased2 = False
    keepdim2 = False
    out2 = (np.array([]), np.array([]))
    input_dict2 = {"input": input2, "dim": dim2, "unbiased": unbiased2, "keepdim": keepdim2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, keepdim = True
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    dim3 = 1
    unbiased3 = True
    keepdim3 = True
    out3 = (np.array([]), np.array([]))
    input_dict3 = {"input": input3, "dim": dim3, "unbiased": unbiased3, "keepdim": keepdim3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor
    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    dim4 = 0
    unbiased4 = True
    keepdim4 = False
    out4 = (np.array([]), np.array([]))
    input_dict4 = {"input": input4, "dim": dim4, "unbiased": unbiased4, "keepdim": keepdim4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))


    # Input 8: Tensor with negative values
    input8 = np.array([-1.0, -2.0, 3.0, 4.0])
    dim8 = 0
    unbiased8 = True
    keepdim8 = False
    out8 = (np.array([]), np.array([]))
    input_dict8 = {"input": input8, "dim": dim8, "unbiased": unbiased8, "keepdim": keepdim8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Different data type (float64)
    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dim9 = 0
    unbiased9 = True
    keepdim9 = False
    out9 = (np.array([]), np.array([]))
    input_dict9 = {"input": input9, "dim": dim9, "unbiased": unbiased9, "keepdim": keepdim9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: larger tensor
    input10 = np.random.rand(5, 5, 5)
    dim10 = 1
    unbiased10 = False
    keepdim10 = True
    out10 = (np.array([]), np.array([]))
    input_dict10 = {"input": input10, "dim": dim10, "unbiased": unbiased10, "keepdim": keepdim10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: 2D tensor, dim=1
    input11 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim11 = 1
    unbiased11 = True
    keepdim11 = False
    out11 = (np.array([]), np.array([]))
    input_dict11 = {"input": input11, "dim": dim11, "unbiased": unbiased11, "keepdim": keepdim11, "out": out11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 15: Check if dim accepts a negative number
    input15 = np.array([1.0, 2.0, 3.0, 4.0])
    dim15 = -1
    unbiased15 = True
    keepdim15 = False
    out15 = (np.array([]), np.array([]))
    input_dict15 = {"input": input15, "dim": dim15, "unbiased": unbiased15, "keepdim": keepdim15, "out": out15}
    list_of_inputs.append(copy.deepcopy(input_dict15))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.std_mean_3"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_3'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_3'], lib="torch", suffix=3)
