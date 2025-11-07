
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lu_solve_inputs():
    list_of_inputs = []
    
    A1 = np.array([[2, 1], [1, 3]], dtype=np.float64)
    b1 = np.array([1, 2], dtype=np.float64)
    LU_pivots1 = np.array([0, 1], dtype=np.int64)
    input_dict1 = {'A': A1, 'b': b1, 'LU_pivots': LU_pivots1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.lu_solve"] = lu_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lu_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_solve'.")


check_valid('torch.lu_solve', generated_inputs['torch.lu_solve'], lib="torch", suffix=0)
