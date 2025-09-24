
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def DoubleStorage_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    list_of_inputs.append({"source": []})

    # Input 2: List of doubles
    list_of_inputs.append({"source": [1.0, 2.0, 3.0]})

    # Input 3: List of floats that can be converted to doubles
    list_of_inputs.append({"source": [1.5, 2.7, 3.9]})

    # Input 4: List of integers that can be converted to doubles
    list_of_inputs.append({"source": [1, 2, 3]})

    # Input 5: List of mixed numbers (ints and floats) that can be converted to doubles
    list_of_inputs.append({"source": [1, 2.5, 3]})

    # Input 6: List with negative numbers
    list_of_inputs.append({"source": [-1.0, 2.0, -3.0]})
    
    # Input 7: List with large numbers
    list_of_inputs.append({"source": [1e10, 2e10, 3e10]})
    
    return list_of_inputs

generated_inputs["torch.DoubleStorage_3"] = DoubleStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.DoubleStorage_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.DoubleStorage_3'.")

check_valid('torch.DoubleStorage', generated_inputs['torch.DoubleStorage_3'], lib="torch")
