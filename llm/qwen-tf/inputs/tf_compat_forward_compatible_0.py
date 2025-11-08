
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_compat_forward_compatible_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(10),
        "day": np.int32(15)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_dict = {
        "year": np.int32(2025),
        "month": np.int32(6),
        "day": np.int32(15)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(2),
        "day": np.int32(29)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - negative values
    input_dict = {
        "year": np.int32(-1),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - zero values
    input_dict = {
        "year": np.int32(0),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - different dimensions
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - large values
    input_dict = {
        "year": np.int32(2050),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - small values
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = tf_compat_forward_compatible_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.forward_compatible' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.forward_compatible'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.compat.forward_compatible', generated_inputs['tf.compat.forward_compatible'], lib="tf", suffix=0)
