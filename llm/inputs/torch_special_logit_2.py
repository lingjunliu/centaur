
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def logit_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with default eps
    input1 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    eps1 = np.array(1e-6, dtype=np.float32)
    input_dict1 = {"input": input1, "eps": eps1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with different eps
    input2 = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    eps2 = np.array(1e-4, dtype=np.float64)
    input_dict2 = {"input": input2, "eps": eps2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with specified eps
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    eps3 = np.array(0.01, dtype=np.float32)
    input_dict3 = {"input": input3, "eps": eps3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with values close to 0 and 1, small eps
    input4 = np.array([0.001, 0.999, 0.5], dtype=np.float64)
    eps4 = np.array(1e-8, dtype=np.float64)
    input_dict4 = {"input": input4, "eps": eps4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with some values outside [0, 1], should be clipped
    input5 = np.array([-0.1, 0.5, 1.1], dtype=np.float32)
    eps5 = np.array(1e-5, dtype=np.float32)
    input_dict5 = {"input": input5, "eps": eps5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Larger eps value
    input6 = np.array([0.2, 0.8], dtype=np.float64)
    eps6 = np.array(0.1, dtype=np.float64)
    input_dict6 = {"input": input6, "eps": eps6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Multi-dimensional array, small eps
    input7 = np.random.rand(5, 5, 5).astype(np.float32)
    eps7 = np.array(1e-7, dtype=np.float32)
    input_dict7 = {"input": input7, "eps": eps7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Float 16
    input8 = np.array([0.3, 0.7], dtype=np.float16)
    eps8 = np.array(1e-3, dtype=np.float16)
    input_dict8 = {"input": input8, "eps": eps8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Scalar tensor
    input9 = np.array(0.6, dtype=np.float32)
    eps9 = np.array(1e-6, dtype=np.float32)
    input_dict9 = {"input": input9, "eps": eps9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: More varied values
    input10 = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64)
    eps10 = np.array(1e-4, dtype=np.float64)
    input_dict10 = {"input": input10, "eps": eps10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.logit_2"] = logit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.logit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.logit_2'.")

check_valid('torch.special.logit', generated_inputs['torch.special.logit_2'], lib="torch", suffix=2)
