
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_IsNan_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 NaN
    x = np.array([np.nan], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 NaN
    x = np.array([np.nan], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Basic half NaN
    x = np.array([np.nan], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 array with NaNs and numbers
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 2D array with NaNs
    x = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half array with positive and negative NaNs
    x = np.array([np.nan, -np.nan], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 array with inf, -inf, and NaN
    x = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 2D array with a name
    x = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "is_nan_check"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half array with multiple nan values
    x = np.array([np.nan, 1.0, 2.0, np.nan, 4.0, np.nan], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsNan"] = tf_raw_ops_IsNan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsNan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsNan'.")

check_valid('tf.raw_ops.IsNan', generated_inputs['tf.raw_ops.IsNan'], lib="tf", suffix=0)
