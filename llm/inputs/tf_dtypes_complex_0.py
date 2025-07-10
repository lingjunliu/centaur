
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dtypes_complex_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    real = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    imag = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    name = "complex_numbers_1"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with float64
    real = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    imag = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float64))
    name = "complex_numbers_2"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    real = tf.constant(np.array([-1.0, -2.0, -3.0], dtype=np.float32))
    imag = tf.constant(np.array([-4.0, -5.0, -6.0], dtype=np.float32))
    name = "complex_numbers_3"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero values
    real = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    imag = tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    name = "complex_numbers_4"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes (2D)
    real = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    imag = tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    name = "complex_numbers_5"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes (3D)
    real = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    imag = tf.constant(np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float32))
    name = "complex_numbers_6"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Empty name
    real = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    imag = tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    name = ""

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero dimension tensor
    real = tf.constant(np.array(1.0, dtype=np.float32))
    imag = tf.constant(np.array(2.0, dtype=np.float32))
    name = "complex_numbers_8"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex float64 values
    real = tf.constant(np.array([1.12345, 2.6789, 3.90123], dtype=np.float64))
    imag = tf.constant(np.array([4.45678, 5.1234, 6.78901], dtype=np.float64))
    name = "complex_numbers_9"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive and negative float64
    real = tf.constant(np.array([-1.5, 2.5, -3.5], dtype=np.float64))
    imag = tf.constant(np.array([4.5, -5.5, 6.5], dtype=np.float64))
    name = "complex_numbers_10"

    input_dict = {
        "real": real,
        "imag": imag,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.dtypes.complex"] = tf_dtypes_complex_inputs()

for i in range(len(generated_inputs["tf.dtypes.complex"])):
    generated_inputs["tf.dtypes.complex"][i]["real"] = generated_inputs["tf.dtypes.complex"][i]["real"].numpy()
    generated_inputs["tf.dtypes.complex"][i]["imag"] = generated_inputs["tf.dtypes.complex"][i]["imag"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.dtypes.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.dtypes.complex'.")

check_valid('tf.dtypes.complex', generated_inputs['tf.dtypes.complex'], lib="tf", suffix=0)
