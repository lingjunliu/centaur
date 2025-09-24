
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sign_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float array with positive, negative and zero values
    x1 = np.array([-2.5, 4.0, 0.0, -0.0, 10.1], dtype=np.float32)
    input_dict_1 = {'x': x1, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D integer array
    x2 = np.array([[-1, 5, 0], [100, -200, -0]], dtype=np.int32)
    input_dict_2 = {'x': x2, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar (0D array) input
    x3 = np.array(-99.9, dtype=np.float64)
    input_dict_3 = {'x': x3, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: int8 array
    x4 = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    input_dict_4 = {'x': x4, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int16 array
    x5 = np.array([[-32768, 32767], [0, -1]], dtype=np.int16)
    input_dict_5 = {'x': x5, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: int64 array
    x6 = np.array([0, -9223372036854775808, 9223372036854775807], dtype=np.int64)
    input_dict_6 = {'x': x6, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D array
    x7 = np.random.uniform(-10, 10, size=(2, 3, 2)).astype(np.float32)
    input_dict_7 = {'x': x7, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex numbers (complex64)
    x8 = np.array([3+4j, -5-12j, 0+0j, 1+0j, 0-1j], dtype=np.complex64)
    input_dict_8 = {'x': x8, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Complex numbers (complex128)
    x9 = np.array([1-2j, -3+4j, 0, 5j], dtype=np.complex128)
    input_dict_9 = {'x': x9, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Special float values (inf, -inf, nan)
    x10 = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    input_dict_10 = {'x': x10, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Empty array
    x11 = np.array([], dtype=np.float32)
    input_dict_11 = {'x': x11, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: All zeros
    x12 = np.zeros((4, 4), dtype=np.float64)
    input_dict_12 = {'x': x12, 'out': None, 'where': None}
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.sign"] = tf_experimental_numpy_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sign'.")

check_valid('tf.experimental.numpy.sign', generated_inputs['tf.experimental.numpy.sign'], lib="tf", suffix=0)
