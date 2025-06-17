
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def DoubleStorage_inputs():
    list_of_inputs = []

    # Input 1: Empty tuple
    input1 = ()
    input_dict1 = {"source": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Tuple of doubles
    input2 = (1.0, 2.0, 3.0)
    input_dict2 = {"source": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tuple of mixed numbers (ints and floats)
    input3 = (1, 2.5, 3, -4.2)
    input_dict3 = {"source": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tuple with a single double
    input4 = (5.0,)
    input_dict4 = {"source": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tuple with negative doubles
    input5 = (-1.0, -2.5, -3.0)
    input_dict5 = {"source": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tuple with zero
    input6 = (0.0, 1.0, 2.0)
    input_dict6 = {"source": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger tuple
    input7 = tuple(np.random.rand(10).tolist())
    input_dict7 = {"source": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.DoubleStorage_4"] = DoubleStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.DoubleStorage_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.DoubleStorage_4'.")

check_valid('torch.DoubleStorage', generated_inputs['torch.DoubleStorage_4'], lib="torch")
