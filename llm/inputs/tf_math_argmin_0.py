
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_argmin_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 10, 26.9, 2.8, 166.32, 62.3], dtype=np.float32)
    axis_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_example_1"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int32
    name = "argmin_example_2"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis_tensor = np.array(1, dtype=np.int32)
    output_type = tf.int32
    name = "argmin_example_3"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis_tensor = np.array(0, dtype=np.int64)
    output_type = tf.int64
    name = "argmin_example_4"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis_tensor = np.array(1, dtype=np.int64)
    output_type = tf.int64
    name = "argmin_example_5"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis_tensor = np.array(2, dtype=np.int64)
    output_type = tf.int64
    name = "argmin_example_6"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([-1, -10, -26.9, -2.8, -166.32, -62.3], dtype=np.float32)
    axis_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int64
    name = "argmin_example_7"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32)
    axis_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int32
    name = "argmin_example_8"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32)
    axis_tensor = np.array(1, dtype=np.int32)
    output_type = tf.int32
    name = "argmin_example_9"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_tensor = np.array([1], dtype=np.int32)
    axis_tensor = np.array(0, dtype=np.int32)
    output_type = tf.int32
    name = "argmin_example_10"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    axis_tensor = np.array(-1, dtype=np.int32)
    output_type = tf.int32
    name = "argmin_example_11"
    input_dict = {"input": input_tensor, "axis": axis_tensor, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.argmin"] = tf_math_argmin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.argmin'.")

check_valid('tf.math.argmin', generated_inputs['tf.math.argmin'], lib="tf", suffix=0)
