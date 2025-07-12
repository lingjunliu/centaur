
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_angle_inputs():
    list_of_inputs = []

    # Input 1, valid
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_1"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input_tensor = np.array([-1-1j, -2-2j, -3-3j], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_2"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    Tout = tf.float64
    name = "angle_3"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input_tensor = np.array([-1-1j, -2-2j, -3-3j], dtype=np.complex128)
    Tout = tf.float64
    name = "angle_4"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, multi-dimensional
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_5"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, multi-dimensional
    input_tensor = np.array([[-1-1j, -2-2j], [-3-3j, -4-4j]], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_6"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, complex numbers with different real and imaginary parts
    input_tensor = np.array([1+2j, 3-1j, -2+1j, -1-3j], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_7"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input_tensor = np.array([1+2j, 3-1j, -2+1j, -1-3j], dtype=np.complex128)
    Tout = tf.float64
    name = "angle_8"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, zero real part
    input_tensor = np.array([0+1j, 0-1j, 0+2j, 0-2j], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_9"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, zero imaginary part
    input_tensor = np.array([1+0j, -1+0j, 2+0j, -2+0j], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_10"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11, valid, with zeros
    input_tensor = np.array([0+0j, 1+1j, -1-1j, 2+0j], dtype=np.complex64)
    Tout = tf.float32
    name = "angle_11"

    input_dict = {
        "input": input_tensor,
        "Tout": Tout,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Angle"] = tf_raw_ops_angle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Angle'.")

check_valid('tf.raw_ops.Angle', generated_inputs['tf.raw_ops.Angle'], lib="tf", suffix=0)
