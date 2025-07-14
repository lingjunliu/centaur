
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def optimized_execution_inputs():
    list_of_inputs = []

    # Input 1: Enabled is True
    input_dict = {"enabled": np.bool_(True).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Enabled is False
    input_dict = {"enabled": np.bool_(False).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Enabled is True (alternative)
    input_dict = {"enabled": np.True_.item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Enabled is False (alternative)
    input_dict = {"enabled": np.False_.item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Enabled is True (casted from int)
    input_dict = {"enabled": np.array(1, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Enabled is False (casted from int)
    input_dict = {"enabled": np.array(0, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Enabled is True (explicit numpy boolean)
    input_dict = {"enabled": np.bool8(True).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Enabled is False (explicit numpy boolean)
    input_dict = {"enabled": np.bool8(False).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Enabled is True (numpy boolean array)
    input_dict = {"enabled": np.array([True], dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Enabled is False (numpy boolean array)
    input_dict = {"enabled": np.array([False], dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.optimized_execution"] = optimized_execution_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.optimized_execution' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.optimized_execution'.")

check_valid('torch.jit.optimized_execution', generated_inputs['torch.jit.optimized_execution'], lib="torch", suffix=0)
