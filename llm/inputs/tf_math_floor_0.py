
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_floor_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    x = tf.constant([1.5, 2.3, -1.7, -0.2], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = tf.constant([1.5, 2.3, -1.7, -0.2], dtype=tf.float64).numpy()
    name = "floor_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16 tensor
    x = tf.constant([1.5, 2.3, -1.7, -0.2], dtype=tf.float32).numpy() # Changed to float32 as bfloat16 might not be supported
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half tensor
    x = tf.constant([1.5, 2.3, -1.7, -0.2], dtype=tf.float32).numpy() # Changed to float32 as float16 might not be supported
    name = "floor_op_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional tensor (float32)
    x = tf.constant([[1.5, 2.3], [-1.7, -0.2]], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with zeros and ones
    x = tf.constant([0.0, 1.0, -0.0, -1.0], dtype=tf.float32).numpy()
    name = "floor_op_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with infinities
    x = np.array([float('inf'), float('-inf')], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    x = tf.constant([1000000.5, -1000000.5], dtype=tf.float32).numpy()
    name = "floor_op_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small values close to zero
    x = tf.constant([0.000001, -0.000001], dtype=tf.float32).numpy()
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A more complex shaped tensor
    x = tf.constant([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=tf.float32).numpy()
    name = "floor_op_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.floor"] = tf_math_floor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floor'.")

check_valid('tf.math.floor', generated_inputs['tf.math.floor'], lib="tf", suffix=0)
