
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_add_inputs():
    list_of_inputs = []

    # Input 1: Basic addition
    x = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    y = tf.constant(np.array([4, 5, 6]), dtype=tf.int32)
    name = "add_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition with negative numbers
    x = tf.constant(np.array([-1, -2, -3]), dtype=tf.int32)
    y = tf.constant(np.array([4, 5, -6]), dtype=tf.int32)
    name = "add_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition with floats
    x = tf.constant(np.array([1.5, 2.5, 3.5]), dtype=tf.float32)
    y = tf.constant(np.array([4.5, 5.5, 6.5]), dtype=tf.float32)
    name = "add_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Addition with different shapes (broadcasting)
    x = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    y = tf.constant(np.array([1, 2]), dtype=tf.int32)
    name = "add_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Addition with higher dimensions
    x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype=tf.int32)
    y = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]), dtype=tf.int32)
    name = "add_example_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition with complex numbers
    x = tf.constant(np.array([1 + 1j, 2 + 2j]), dtype=tf.complex64)
    y = tf.constant(np.array([3 + 3j, 4 + 4j]), dtype=tf.complex64)
    name = "add_example_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition with uint8
    x = tf.constant(np.array([1, 2, 3], dtype=np.uint8))
    y = tf.constant(np.array([4, 5, 6], dtype=np.uint8))
    name = "add_example_7"
    x = tf.cast(x, tf.uint8)
    y = tf.cast(y, tf.uint8)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Addition with int64
    x = tf.constant(np.array([1, 2, 3], dtype=np.int64))
    y = tf.constant(np.array([4, 5, 6], dtype=np.int64))
    name = "add_example_8"
    x = tf.cast(x, tf.int64)
    y = tf.cast(y, tf.int64)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Addition with half
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    y = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float16))
    name = "add_example_9"
    x = tf.cast(x, tf.float16)
    y = tf.cast(y, tf.float16)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Addition with float64
    x = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=np.float64)
    y = tf.constant(np.array([4.0, 5.0, 6.0]), dtype=np.float64)
    name = "add_example_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.add"] = tf_math_add_inputs()
for k,v in generated_inputs.items():
    for i in range(len(v)):
        generated_inputs[k][i]['x'] = generated_inputs[k][i]['x'].numpy()
        generated_inputs[k][i]['y'] = generated_inputs[k][i]['y'].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.add'.")

check_valid('tf.math.add', generated_inputs['tf.math.add'], lib="tf", suffix=0)
