
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_add_n_inputs():
    list_of_inputs = []

    def create_input_dict(inputs, name=None):
        return {"inputs": inputs, "name": name}

    # Input 1: Basic case with two tensors
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 2: Three tensors, different values
    a = np.array([[1, -2], [-3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, -8]], dtype=np.int32)
    c = np.array([[-9, 10], [11, 12]], dtype=np.int32)
    inputs = [tf.constant(a), tf.constant(b), tf.constant(c)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "sum_of_three")))

    # Input 3: Tensors with float32 type
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    b = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float32)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 4: Tensors with float64 type
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    b = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float64)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 5: Three dimensional tensors
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 6: One tensor
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    inputs = [tf.constant(a)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 7: Several tensors, all zeros
    a = np.zeros((2, 2), dtype=np.int32)
    b = np.zeros((2, 2), dtype=np.int32)
    c = np.zeros((2, 2), dtype=np.int32)
    inputs = [tf.constant(a), tf.constant(b), tf.constant(c)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 8: Tensors with int64 type
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    b = np.array([[5, 6], [7, 8]], dtype=np.int64)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 9: Tensors with negative values
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    b = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 10: Simple vector addition
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 11: Different shapes but same length (should throw error in TF) - Added it as a potential test
    a = np.array([1, 2, 3, 4], dtype=np.int32).reshape((2, 2))
    b = np.array([5, 6, 7, 8], dtype=np.int32).reshape((4, 1))
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    # Input 12: Broadcasting not supported - Test
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([1, 2], dtype=np.int32)
    inputs = [tf.constant(a), tf.constant(b)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs)))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.add_n"] = tf_math_add_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.add_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.add_n'.")

check_valid('tf.math.add_n', generated_inputs['tf.math.add_n'], lib="tf", suffix=0)
