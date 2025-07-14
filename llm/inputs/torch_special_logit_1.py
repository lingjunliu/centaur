
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def logit_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    eps1 = 1e-6
    input_dict1 = {"input": input1, "eps": eps1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with values close to 0 and 1
    input2 = np.array([[0.001, 0.999], [0.1, 0.9]], dtype=np.float64)
    eps2 = 1e-8
    input_dict2 = {"input": input2, "eps": eps2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tensor with a small epsilon
    input3 = np.array([0.5], dtype=np.float32)
    eps3 = 1e-12
    input_dict3 = {"input": input3, "eps": eps3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 3D tensor
    input4 = np.random.rand(2, 3, 4).astype(np.float32)
    eps4 = 1e-5
    input_dict4 = {"input": input4, "eps": eps4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with large epsilon
    input5 = np.array([0.3, 0.7], dtype=np.float64)
    eps5 = 0.1
    input_dict5 = {"input": input5, "eps": eps5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Values spanning the range (0, 1)
    input6 = np.linspace(0.01, 0.99, 5).astype(np.float32)
    eps6 = 1e-7
    input_dict6 = {"input": input6, "eps": eps6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Empty tensor
    input7 = np.array([], dtype=np.float64)
    eps7 = 1e-9
    input_dict7 = {"input": input7, "eps": eps7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Single element tensor
    input8 = np.array([0.6], dtype=np.float32)
    eps8 = 1e-4
    input_dict8 = {"input": input8, "eps": eps8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Tensor with different dtype (float16)
    input9 = np.array([0.4, 0.6], dtype=np.float16)
    eps9 = 1e-3
    input_dict9 = {"input": input9, "eps": eps9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: 4D tensor
    input10 = np.random.rand(2, 2, 2, 2).astype(np.float64)
    eps10 = 1e-8
    input_dict10 = {"input": input10, "eps": eps10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Tensor with repeated values
    input11 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    eps11 = 1e-6
    input_dict11 = {"input": input11, "eps": eps11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: Large tensor
    input12 = np.random.rand(1000).astype(np.float64)
    eps12 = 1e-7
    input_dict12 = {"input": input12, "eps": eps12}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.logit_1"] = logit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.logit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.logit_1'.")

check_valid('torch.special.logit', generated_inputs['torch.special.logit_1'], lib="torch", suffix=1)
