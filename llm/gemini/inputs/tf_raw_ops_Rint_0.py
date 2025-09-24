
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_rint_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = np.float32(1.5)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 scalar, negative
    x = np.float32(-2.5)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 array
    x = np.array([1.2, 2.7, -3.1, -4.9], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 array, multi-dimensional
    x = np.array([[0.5, 1.5], [-1.5, -0.5]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 scalar
    x = np.float64(3.5)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 array
    x = np.array([0.1, 1.9, -2.3, -3.7], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half scalar
    x = np.float16(2.5)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half array
    x = np.array([0.6, 1.4, -2.6, -3.4], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16 scalar
    x = np.float32(4.5) # bfloat16 doesn't exist in numpy, using float32 and hope it will be casted to bfloat16 in tf
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 array
    x = np.array([0.7, 1.3, -2.7, -3.3], dtype=np.float32) # bfloat16 doesn't exist in numpy, using float32 and hope it will be casted to bfloat16 in tf
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Rint"] = tf_raw_ops_rint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Rint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Rint'.")

check_valid('tf.raw_ops.Rint', generated_inputs['tf.raw_ops.Rint'], lib="tf", suffix=0)
