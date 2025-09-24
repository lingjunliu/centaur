
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_real_inputs():
    list_of_inputs = []

    # Input 1: Simple complex tensor
    input_tensor = tf.constant([1 + 2j, 3 + 4j, 5 + 6j], dtype=tf.complex64).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Real tensor
    input_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    input_dict = {"input": input_tensor, "name": "real_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor with negative values
    input_tensor = tf.constant([-1 - 2j, -3 + 4j, 5 - 6j], dtype=tf.complex128).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Real tensor with negative values
    input_tensor = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float64).numpy()
    input_dict = {"input": input_tensor, "name": "negative_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D complex tensor
    input_tensor = tf.constant([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=tf.complex64).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D real tensor
    input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    input_dict = {"input": input_tensor, "name": "2d_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D complex tensor
    input_tensor = tf.constant([[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], [[9 + 10j, 11 + 12j], [13 + 14j, 15 + 16j]]], dtype=tf.complex128).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor
    input_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "empty_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex tensor with zero values
    input_tensor = tf.constant([0 + 0j, 0 + 0j, 0 + 0j], dtype=tf.complex64).numpy()
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Real tensor with zero values
    input_tensor = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32).numpy()
    input_dict = {"input": input_tensor, "name": "zero_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.real"] = tf_math_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.real'.")

check_valid('tf.math.real', generated_inputs['tf.math.real'], lib="tf", suffix=0)
