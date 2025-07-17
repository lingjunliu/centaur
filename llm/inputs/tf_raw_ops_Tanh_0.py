
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_tanh_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32).numpy(), "name": "tanh_float32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64).numpy(), "name": "tanh_float64_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, scalar.  Using float16 instead of bfloat16
    x = np.array(0.5, dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16).numpy(), "name": "tanh_float16_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16).numpy(), "name": "tanh_half_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D array
    x = np.array([1 + 1j, 2 - 2j, -1 + 0j], dtype=np.complex64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex64).numpy(), "name": "tanh_complex64_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D array
    x = np.array([[1 + 1j, 2 - 2j], [-1 + 0j, 0 - 1j]], dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex128).numpy(), "name": "tanh_complex128_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, large values
    x = np.array([-100.0, 100.0, -1000.0, 1000.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32).numpy(), "name": "tanh_float32_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, mixed positive and negative values
    x = np.array([-0.8, -0.2, 0.3, 0.9], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16).numpy(), "name": "tanh_half_mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, zero array
    x = np.zeros((3, 3), dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64).numpy(), "name": "tanh_float64_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16, array with varied values. Using float16 instead of bfloat16
    x = np.array([-0.2, 0.0, 0.3, 1.0, 5.0], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16).numpy(), "name": "tanh_float16_varied"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_tanh_inputs()
generated_inputs["tf.raw_ops.Tanh"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Tanh'.")

check_valid('tf.raw_ops.Tanh', generated_inputs['tf.raw_ops.Tanh'], lib="tf", suffix=0)
