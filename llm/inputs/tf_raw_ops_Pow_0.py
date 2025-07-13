
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_pow_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 0.5], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32),
                  "y": tf.convert_to_tensor(y, dtype=tf.float32),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 with negative exponents
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([-2.0, 3.0, -0.5], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32),
                  "y": tf.convert_to_tensor(y, dtype=tf.float32),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16
    x = np.array([2.0, 3.0], dtype=np.float16)
    y = np.array([2.0, 3.0], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16),
                  "y": tf.convert_to_tensor(y, dtype=tf.float16),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64
    x = np.array([2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 3.0], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64),
                  "y": tf.convert_to_tensor(y, dtype=tf.float64),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int8 (converted to float32)
    x = np.array([2, 3], dtype=np.int8).astype(np.float32)
    y = np.array([2, 3], dtype=np.int8).astype(np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32),
                  "y": tf.convert_to_tensor(y, dtype=tf.float32),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16 (converted to float32)
    x = np.array([2, 3], dtype=np.int16).astype(np.float32)
    y = np.array([2, 3], dtype=np.int16).astype(np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32),
                  "y": tf.convert_to_tensor(y, dtype=tf.float32),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 (converted to float32)
    x = np.array([2, 3], dtype=np.int64).astype(np.float32)
    y = np.array([2, 3], dtype=np.int64).astype(np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32),
                  "y": tf.convert_to_tensor(y, dtype=tf.float32),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64
    x = np.array([2 + 1j, 3 + 2j], dtype=np.complex64)
    y = np.array([2 + 0j, 3 - 1j], dtype=np.complex64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex64),
                  "y": tf.convert_to_tensor(y, dtype=tf.complex64),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128
    x = np.array([2 + 1j, 3 + 2j], dtype=np.complex128)
    y = np.array([2 + 0j, 3 - 1j], dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex128),
                  "y": tf.convert_to_tensor(y, dtype=tf.complex128),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different shapes, broadcasting
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32),
                  "y": tf.convert_to_tensor(y, dtype=tf.float32),
                  "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pow'.")

check_valid('tf.raw_ops.Pow', generated_inputs['tf.raw_ops.Pow'], lib="tf", suffix=0)
