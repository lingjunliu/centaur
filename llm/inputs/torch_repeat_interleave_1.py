
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def repeat_interleave_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive repeats, dim=0
    input1 = np.array([1, 2, 3])
    repeats1 = 2
    dim1 = 0
    input_dict1 = {"input": input1, "repeats": repeats1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, positive repeats, dim=0
    input2 = np.array([[1, 2], [3, 4]])
    repeats2 = 3
    dim2 = 0
    input_dict2 = {"input": input2, "repeats": repeats2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, positive repeats, dim=1
    input3 = np.array([[1, 2], [3, 4]])
    repeats3 = 2
    dim3 = 1
    input_dict3 = {"input": input3, "repeats": repeats3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor repeats as numpy array
    input4 = np.array([1, 2, 3])
    repeats4 = np.array([2,1,2])
    dim4 = 0
    input_dict4 = {"input": input4, "repeats": repeats4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor, positive repeats, dim=1 repeats as numpy array
    input5 = np.array([[1, 2], [3, 4]])
    repeats5 = np.array([1,2])
    dim5 = 0
    input_dict5 = {"input": input5, "repeats": repeats5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D tensor, repeats=1
    input6 = np.array([1, 2, 3])
    repeats6 = 1
    dim6 = 0
    input_dict6 = {"input": input6, "repeats": repeats6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 2D tensor repeats as numpy array, dim=1
    input7 = np.array([[1, 2], [3, 4]])
    repeats7 = np.array([2,1])
    dim7 = 1
    input_dict7 = {"input": input7, "repeats": repeats7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D tensor, positive repeats, dim=0
    input8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    repeats8 = 2
    dim8 = 0
    input_dict8 = {"input": input8, "repeats": repeats8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 3D tensor, positive repeats, dim=1
    input9 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    repeats9 = 2
    dim9 = 1
    input_dict9 = {"input": input9, "repeats": repeats9, "dim": dim9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: 1D tensor, repeats as scalar
    input10 = np.array([1, 2, 3])
    repeats10 = np.int64(2)
    dim10 = 0
    input_dict10 = {"input": input10, "repeats": repeats10, "dim": dim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.repeat_interleave_1"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.repeat_interleave_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_1'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_1'], lib="torch", suffix=1)
