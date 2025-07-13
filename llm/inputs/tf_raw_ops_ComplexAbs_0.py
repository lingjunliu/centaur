
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_complex_abs_inputs():
    list_of_inputs = []

    # Input 1: Basic complex64
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    Tout = tf.float32
    name = "complex_abs_1"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic complex128
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    Tout = tf.float64
    name = "complex_abs_2"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex64 with negative values
    x = np.array([-1 - 1j, -2 + 2j, 3 - 3j], dtype=np.complex64)
    Tout = tf.float32
    name = "complex_abs_3"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex128 with a different name
    x = np.array([-1 - 1j, -2 + 2j, 3 - 3j], dtype=np.complex128)
    Tout = tf.float64
    name = "my_complex_abs_4"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64, multi-dimensional array
    x = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    Tout = tf.float32
    name = "complex_abs_5"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128, multi-dimensional array
    x = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)
    Tout = tf.float64
    name = "complex_abs_6"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64, all zeros
    x = np.array([0 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex64)
    Tout = tf.float32
    name = "complex_abs_7"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128, all zeros
    x = np.array([0 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex128)
    Tout = tf.float64
    name = "complex_abs_8"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex64, large values
    x = np.array([1000 + 1000j, -2000 + 2000j, 3000 - 3000j], dtype=np.complex64)
    Tout = tf.float32
    name = "complex_abs_9"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex128, large values
    x = np.array([1000 + 1000j, -2000 + 2000j, 3000 - 3000j], dtype=np.complex128)
    Tout = tf.float64
    name = "complex_abs_10"
    input_dict = {"x": x, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ComplexAbs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ComplexAbs'.")

check_valid('tf.raw_ops.ComplexAbs', generated_inputs['tf.raw_ops.ComplexAbs'], lib="tf", suffix=0)
