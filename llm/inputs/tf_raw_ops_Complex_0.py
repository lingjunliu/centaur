
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_complex_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 complex64
    real_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    imag_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex64
    name = "complex_example_1"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 complex128
    real_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    imag_np = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex128
    name = "complex_example_2"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array, float32 complex64
    real_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    imag_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex64
    name = "complex_example_3"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array, float64 complex128
    real_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    imag_np = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float64)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex128
    name = "complex_example_4"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values, float32, complex64
    real_np = np.array([-1.0, -2.0, 3.0], dtype=np.float32)
    imag_np = np.array([4.0, -5.0, -6.0], dtype=np.float32)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex64
    name = "complex_example_5"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values, float64, complex128
    real_np = np.array([-1.0, -2.0, 3.0], dtype=np.float64)
    imag_np = np.array([4.0, -5.0, -6.0], dtype=np.float64)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex128
    name = "complex_example_6"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7:  float32, default Tout
    real_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    imag_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex64
    name = "complex_example_7"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, default Tout
    real_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    imag_np = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex128
    name = "complex_example_8"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero values, float32, complex64
    real_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    imag_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex64
    name = "complex_example_9"
    input_dict = {"real": real, "imag": imag, "Tout": Tout, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, complex128, different values
    real_np = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    imag_np = np.array([4.4, 5.5, 6.6], dtype=np.float64)
    real = tf.constant(real_np)
    imag = tf.constant(imag_np)
    Tout = tf.complex128
    name = "complex_example_10"
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
