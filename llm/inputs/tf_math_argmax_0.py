
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_argmax_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 5, 2, 8, 3])
    axis = 0
    output_type = tf.int64
    name = "argmax_example_1"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 5, 2], [8, 3, 9]])
    axis = 0
    output_type = tf.int64
    name = "argmax_example_2"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 5, 2], [8, 3, 9]])
    axis = 1
    output_type = tf.int64
    name = "argmax_example_3"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 5], [2, 8]], [[3, 9], [4, 7]]])
    axis = 0
    output_type = tf.int64
    name = "argmax_example_4"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 5], [2, 8]], [[3, 9], [4, 7]]])
    axis = 1
    output_type = tf.int64
    name = "argmax_example_5"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 5], [2, 8]], [[3, 9], [4, 7]]])
    axis = 2
    output_type = tf.int64
    name = "argmax_example_6"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - negative values
    input_tensor = np.array([-1, -5, -2, -8, -3])
    axis = 0
    output_type = tf.int64
    name = "argmax_example_7"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - all same values
    input_tensor = np.array([5, 5, 5, 5, 5])
    axis = 0
    output_type = tf.int64
    name = "argmax_example_8"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Different output type
    input_tensor = np.array([1, 5, 2, 8, 3])
    axis = 0
    output_type = tf.int32
    name = "argmax_example_9"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Multi dimensional with negative values
    input_tensor = np.array([[-1, 5, -2], [8, -3, 9]])
    axis = 0
    output_type = tf.int64
    name = "argmax_example_10"
    input_dict = {"input": input_tensor, "axis": axis, "output_type": output_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.argmax"] = tf_math_argmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.argmax'.")

check_valid('tf.math.argmax', generated_inputs['tf.math.argmax'], lib="tf", suffix=0)
