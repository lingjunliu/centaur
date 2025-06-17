
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ndtr_inputs():
    list_of_inputs = []

    input_1 = np.array(0.0)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.array(1.0)
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = np.array(-1.0)
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = np.array([0.0, 0.5, 1.0])
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = np.array([[-1.0, 0.0], [0.5, 1.0]])
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    return list_of_inputs

generated_inputs["torch.special.ndtr"] = ndtr_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.ndtr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.ndtr'.")

check_valid('torch.special.ndtr', generated_inputs['torch.special.ndtr'], lib="torch")
