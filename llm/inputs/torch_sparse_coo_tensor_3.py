
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D sparse tensor
    # size = (3, 4)
    # dtype = np.float32
    # requires_grad = False
    # input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D sparse tensor with different dtype
    # size = (2, 3, 5)
    # dtype = np.int64
    # requires_grad = True
    # input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D sparse tensor
    #size = (10,)
    #dtype = np.float64
    #requires_grad = False
    #input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty sparse tensor
    #size = (0, 5)
    #dtype = np.float32
    #requires_grad = True
    #input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger size
    size = (100, 200)
    dtype = np.float16
    requires_grad = False
    input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean type
    #size = (5, 5)
    #dtype = np.bool_
    #requires_grad = True
    #input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int8 type
    #size = (4, 4)
    #dtype = np.int8
    #requires_grad = False
    #input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Uint8 type
    #size = (6, 6)
    #dtype = np.uint8
    #requires_grad = True
    #input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    #list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor
    size = (2, 3, 4, 5)
    dtype = np.float32
    requires_grad = False
    input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small size, different dtype
    #size = (2, 2)
    #dtype = np.int32
    #requires_grad = True
    #input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Another different size and type
    #size = (7, 8)
    #dtype = np.float64
    #requires_grad = False
    #input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    size = (1,1)
    dtype = np.float32
    requires_grad = False
    input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    size = (10, 1)
    dtype = np.float32
    requires_grad = True
    input_dict = {"size": size, "dtype": dtype, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sparse_coo_tensor_3"] = sparse_coo_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_coo_tensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_3'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_3'], lib="torch", suffix=3)
