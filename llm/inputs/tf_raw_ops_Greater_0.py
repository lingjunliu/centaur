
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_greater_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32
    x = np.array([5, 4, 6], dtype=np.int32)
    y = np.array([5, 2, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "greater_int32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcasting with int32
    x = np.array([5, 4, 6], dtype=np.int32)
    y = np.array([5], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "greater_int32_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values with int32
    x = np.array([-1, 0, 1], dtype=np.int32)
    y = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "greater_int32_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float32 with different shapes
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "greater_float32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 with negative values
    x = np.array([-1.5, -0.5, 0.5], dtype=np.float64)
    y = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "greater_float64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64
    x = np.array([5, 4, 6], dtype=np.int64)
    y = np.array([5, 2, 5], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "greater_int64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_greater_inputs()
generated_inputs["tf.raw_ops.Greater"] = []
for input_dict in inputs:
    x = tf.convert_to_tensor(input_dict["x"])
    y = tf.convert_to_tensor(input_dict["y"])
    generated_inputs["tf.raw_ops.Greater"].append({"x": x,
                                                     "y": y,
                                                     "name": input_dict["name"]})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Greater' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Greater'.")

check_valid('tf.raw_ops.Greater', generated_inputs['tf.raw_ops.Greater'], lib="tf", suffix=0)
