
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isfinite_inputs():
    list_of_inputs = []

    # Input 1: Basic finite values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x.astype(np.float32), "name": "finite_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Mixed finite and infinite values
    x = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    input_dict = {"x": x.astype(np.float32), "name": "mixed_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed finite and NaN values
    x = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    input_dict = {"x": x.astype(np.float32), "name": "nan_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Only infinite values
    x = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    input_dict = {"x": x.astype(np.float32), "name": "inf_only"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Only NaN values
    x = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    input_dict = {"x": x.astype(np.float32), "name": "nan_only"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array
    x = np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float32)
    input_dict = {"x": x.astype(np.float32), "name": "multi_dim"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 type
    x = np.array([1.0, 2.0, np.inf], dtype=np.float16)
    input_dict = {"x": x.astype(np.float16), "name": "bfloat16_type"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 type
    x = np.array([1.0, 2.0, np.nan], dtype=np.float64)
    input_dict = {"x": x.astype(np.float64), "name": "float64_type"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half type
    x = np.array([1.0, 2.0, np.inf], dtype=np.float16)
    input_dict = {"x": x.astype(np.float16), "name": "half_type"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x.astype(np.float32), "name": "all_zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsFinite"] = tf_raw_ops_isfinite_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsFinite' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsFinite'.")

check_valid('tf.raw_ops.IsFinite', generated_inputs['tf.raw_ops.IsFinite'], lib="tf", suffix=0)
