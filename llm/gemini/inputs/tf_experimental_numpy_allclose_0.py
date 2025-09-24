
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_allclose_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.constant([1.001, 2.002, 3.003], dtype=tf.float32)
    rtol = 1e-02
    atol = 1e-03
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant([1.0, 2.0, np.nan], dtype=tf.float32)
    b = tf.constant([1.0, 2.0, np.nan], dtype=tf.float32)
    rtol = 1e-05
    atol = 1e-08
    equal_nan = True
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    b = tf.constant([[1.0, 2.0], [3.0, 4.0000001]], dtype=tf.float32)
    rtol = 1e-07
    atol = 0.0
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant([1, 2, 3], dtype=tf.int32)
    b = tf.constant([1, 2, 4], dtype=tf.int32)
    rtol = 1e-05
    atol = 1
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": 1e-05, "atol": 1.0, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.constant([1.1, 2.2, 3.3], dtype=tf.float32)
    rtol = 0.1
    atol = 0.1
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=tf.float32)
    b = tf.constant(np.array([[1.000001, 2.0], [3.0, 4.000001]]), dtype=tf.float32)
    rtol = 1e-06
    atol = 0.0
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([np.inf, -np.inf, 1.0]), dtype=tf.float32)
    b = tf.constant(np.array([np.inf, -np.inf, 1.0000001]), dtype=tf.float32)
    rtol = 1e-07
    atol = 0.0
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.constant([1.0, 2.0, np.nan], dtype=tf.float32)
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.constant([1.0 + 1e-09, 2.0, 3.0], dtype=tf.float32)
    rtol = 1e-08
    atol = 0.0
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    a = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    b = tf.constant(np.array([1.0, 2.0, 3.0]), dtype=tf.float32)
    rtol = np.float32(1e-05)
    atol = np.float32(1e-08)
    equal_nan = False
    input_dict = {"a": a, "b": b, "rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.allclose"] = tf_experimental_numpy_allclose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.allclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.allclose'.")

check_valid('tf.experimental.numpy.allclose', generated_inputs['tf.experimental.numpy.allclose'], lib="tf", suffix=0)
