
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_all_inputs():
    list_of_inputs = []

    # Input 1: Bool tensor with all True
    input1 = np.array([[True, True], [True, True]])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Bool tensor with some False
    input2 = np.array([[True, False], [True, True]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Int tensor, all non-zero (evaluates to True)
    input3 = np.array([[1, 2], [3, 4]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Int tensor with zero (evaluates to False)
    input4 = np.array([[1, 0], [3, 4]])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor, all non-zero (evaluates to True)
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float tensor with zero (evaluates to False)
    input6 = np.array([[1.0, 0.0], [3.0, 4.0]])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 3D Bool tensor
    input7 = np.array([[[True, True], [True, True]], [[True, True], [True, True]]])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Empty tensor
    input8 = np.array([])
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Negative values (int)
    input9 = np.array([[-1, -2], [-3, -4]])
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    return list_of_inputs

generated_inputs["torch.all_1"] = torch_all_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.all_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_1'.")

check_valid('torch.all', generated_inputs['torch.all_1'], lib="torch")
