
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isnan_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 array with NaN
    x = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 array with multiple NaNs and infinities
    x = np.array([np.inf, np.nan, -np.inf, np.nan, 0.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test_isnan"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16 array - converted to float32 for numpy array
    x = np.array([1.0, np.nan, 3.0], dtype=np.float16)
    x = x.astype(np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: half array
    x = np.array([1.0, np.nan, 3.0], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Multi-dimensional float32 array
    x = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values and NaN in float64 array
    x = np.array([-1.0, np.nan, -3.5, np.nan], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty float32 array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 array with only NaN values
    x = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large float64 array
    x = np.random.rand(100, 100).astype(np.float64)
    x[50][50] = np.nan
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array
    x = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "3d_array"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsNan"] = tf_raw_ops_isnan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsNan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsNan'.")

check_valid('tf.raw_ops.IsNan', generated_inputs['tf.raw_ops.IsNan'], lib="tf", suffix=0)
