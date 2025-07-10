
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isclose_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    b = tf.constant([1.0, 2.0, 3.1], dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64).numpy()
    b = tf.constant([[1.0, 2.1], [3.2, 4.0]], dtype=tf.float64).numpy()
    rtol = 1e-03
    atol = 1e-05
    equal_nan = True
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant([np.nan, 2.0, 3.0], dtype=tf.float32).numpy()
    b = tf.constant([np.nan, 2.0, 3.0], dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-08
    equal_nan = True
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    b = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    b = tf.constant([1.1, 2.1, 3.1], dtype=tf.float32).numpy()
    rtol = 0.1
    atol = 0.0
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    b = tf.constant([1.0, 2.0, 3.0 + 1e-7], dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-07
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant([1.0, 2.0, np.inf], dtype=tf.float32).numpy()
    b = tf.constant([1.0, 2.0, np.inf], dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant([1.0, 2.0, -3.0], dtype=tf.float32).numpy()
    b = tf.constant([1.0, 2.0, -3.1], dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant(1.0, dtype=tf.float32).numpy()
    b = tf.constant(1.0 + 1e-6, dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float32).numpy()
    b = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0 + 1e-7]]], dtype=tf.float32).numpy()
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isclose"] = tf_experimental_numpy_isclose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.isclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isclose'.")

check_valid('tf.experimental.numpy.isclose', generated_inputs['tf.experimental.numpy.isclose'], lib="tf", suffix=0)
