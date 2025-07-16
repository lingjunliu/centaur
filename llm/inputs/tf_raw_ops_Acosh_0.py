
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_acosh_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 with values >= 1
    x = np.array([1.0, 1.5, 2.0, 10.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with values >= 1
    x = np.array([1.0, 2.718, 3.14159, 100.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "acosh_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16 with values >= 1
    x = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32).astype(np.float16)
    x = tf.cast(tf.constant(x), dtype=tf.bfloat16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half (float16) with values >= 1
    x = np.array([1.0, 1.25, 1.5, 1.75], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": "acosh_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 with values >= 1 (real part)
    x = np.array([1.0 + 0j, 2.0 + 1j, 3.0 - 2j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 with values >= 1 (real part)
    x = np.array([1.0 + 0j, 2.0 + 1j, 3.0 - 2j], dtype=np.complex128)
    input_dict = {"x": tf.constant(x), "name": "acosh_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional array (float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another multi-dimensional array (float64)
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "acosh_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 with large values
    x = np.array([1e5, 1e6, 1e7], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 with a mix of small and large values >= 1
    x = np.array([1.0, 1.000001, 1000000.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "acosh_example_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: half (float16) values >=1
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    input_dict = {"x": x, "name": "acosh_example_11"}
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
