
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_asinh_inputs():
    list_of_inputs = []

    # Input 3: float32
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64
    x = np.array([-100.0, -0.1, 0.1, 100.0], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "asinh_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array([-1.0 + 1j, 0.0 - 1j, 1.0 + 0j], dtype=np.complex64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = np.array([-2.0 - 2j, 0.5 + 0j, 2.0 - 0.5j], dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex128), "name": "asinh_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: multi-dimensional float32
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large values float32
    x = np.array([-1e5, 1e5], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: small values float32
    x = np.array([-1e-5, 1e-5], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with different shape
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Asinh"] = tf_raw_ops_asinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Asinh'.")

check_valid('tf.raw_ops.Asinh', generated_inputs['tf.raw_ops.Asinh'], lib="tf", suffix=0)
