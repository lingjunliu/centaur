
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import datetime

def tf_compat_forward_compatible_inputs():
    list_of_inputs = []

    # Input 1: A date in the past
    input_dict = {
        "year": np.int32(2020),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A date in the future
    input_dict = {
        "year": np.int32(2025),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Today's date
    today = datetime.date.today()
    input_dict = {
        "year": np.int32(today.year),
        "month": np.int32(today.month),
        "day": np.int32(today.day)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Edge case - last day of the year
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Edge case - first day of the year
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Edge case - leap year
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(2),
        "day": np.int32(29)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another date in the past
    input_dict = {
        "year": np.int32(2015),
        "month": np.int32(6),
        "day": np.int32(15)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another date in the future
    input_dict = {
        "year": np.int32(2030),
        "month": np.int32(7),
        "day": np.int32(20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Month boundary
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(3),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Day boundary
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(1),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = tf_compat_forward_compatible_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.forward_compatible' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.forward_compatible'.")

check_valid('tf.compat.forward_compatible', generated_inputs['tf.compat.forward_compatible'], lib="tf", suffix=0)
