
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_round_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = np.float32(2.3)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 array
    x = np.array([-2.5, -1.5, 0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 array
    x = np.array([-2.5, -1.5, 0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 matrix
    x = np.array([[1.2, 2.7], [3.5, 4.1]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 matrix with negative values
    x = np.array([[-1.2, -2.7], [-3.5, -4.1]], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32 array
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 array
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16 array
    x = np.array([-2.5, -1.5, 0.5, 1.5, 2.5], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Round"] = tf_raw_ops_round_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Round'.")

check_valid('tf.raw_ops.Round', generated_inputs['tf.raw_ops.Round'], lib="tf", suffix=0)
