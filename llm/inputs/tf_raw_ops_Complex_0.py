
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_complex_inputs():
    list_of_inputs = []

    # Input 1
    real = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    imag = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    Tout = tf.complex64
    name = "complex_numbers_1"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    imag = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    Tout = tf.complex64
    name = "complex_numbers_2"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    real = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    imag = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    Tout = tf.complex128
    name = "complex_numbers_3"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    imag = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    Tout = tf.complex128
    name = "complex_numbers_4"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    real = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    imag = np.array([-4.0, -5.0, -6.0], dtype=np.float32)
    Tout = tf.complex64
    name = "complex_numbers_5"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    real = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    imag = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    Tout = tf.complex64
    name = "complex_numbers_6"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    real = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    imag = np.array([4.5, 5.5, 6.5], dtype=np.float32)
    Tout = tf.complex64
    name = "complex_numbers_7"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    real = np.array([1.0], dtype=np.float32)
    imag = np.array([4.0], dtype=np.float32)
    Tout = tf.complex64
    name = "complex_numbers_8"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    real = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    imag = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float64)
    Tout = tf.complex128
    name = "complex_numbers_9"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    real = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    imag = np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    Tout = tf.complex64
    name = "complex_numbers_10"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Complex"] = tf_raw_ops_complex_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Complex'.")

check_valid('tf.raw_ops.Complex', generated_inputs['tf.raw_ops.Complex'], lib="tf", suffix=0)
