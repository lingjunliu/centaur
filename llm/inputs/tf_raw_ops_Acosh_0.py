
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_acosh_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, positive values >= 1
    x = np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x.astype(np.float32)), "name": "acosh_test_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array, positive values >= 1
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": tf.constant(x.astype(np.float64)), "name": "acosh_test_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, scalar value, 1
    x = np.array(1.0, dtype=np.float16)
    input_dict = {"x": tf.constant(x.astype(np.float16)), "name": "acosh_test_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16, 1D array, mixed values >= 1
    x = np.array([1.0, 2.5, 10.2], dtype=np.float16)
    input_dict = {"x": tf.constant(x.astype(np.float16)), "name": "acosh_test_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array, values >= 1, real part only.
    x = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x.astype(np.complex64)), "name": "acosh_test_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D array, values >= 1, real part only.
    x = np.array([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], dtype=np.complex128)
    input_dict = {"x": tf.constant(x.astype(np.complex128)), "name": "acosh_test_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": tf.constant(x.astype(np.float32)), "name": "acosh_test_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, scalar
    x = np.array(5.5, dtype=np.float64)
    input_dict = {"x": tf.constant(x.astype(np.float64)), "name": "acosh_test_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16, 2D array
    x = np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float16)
    input_dict = {"x": tf.constant(x.astype(np.float16)), "name": "acosh_test_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16, 3D array
    x = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float16)
    input_dict = {"x": tf.constant(x.astype(np.float16)), "name": "acosh_test_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float32 with large value
    x = np.array([1e10], dtype=np.float32)
    input_dict = {"x": tf.constant(x.astype(np.float32)), "name": "acosh_test_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Acosh"] = tf_raw_ops_acosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Acosh'.")

check_valid('tf.raw_ops.Acosh', generated_inputs['tf.raw_ops.Acosh'], lib="tf", suffix=0)
