
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sin_inputs():
    list_of_inputs = []

    # Input 1: float32 tensor
    x = tf.constant(np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], dtype=np.float32)).numpy()
    name = "sin_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = tf.constant(np.array([-1.0, 1.0, 2.0, -2.0], dtype=np.float64)).numpy()
    name = "sin_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half tensor
    x = tf.constant(np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float16)).numpy()
    name = "sin_half"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16 tensor - Removing this because of dtype issues in the checker
    #x = tf.constant(np.array([0.25, -0.75, 1.25, -1.75], dtype=tf.bfloat16.as_numpy_dtype)).numpy()
    #name = "sin_bfloat16"
    #input_dict = {"x": x, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 tensor
    x = tf.constant(np.array([1+1j, 2-2j, -1+2j, -2-1j], dtype=np.complex64)).numpy()
    name = "sin_complex64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 tensor
    x = tf.constant(np.array([1.5+0.5j, 2.5-1.5j, -0.5+1.5j, -1.5-2.5j], dtype=np.complex128)).numpy()
    name = "sin_complex128"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional float32 tensor
    x = tf.constant(np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]], dtype=np.float32)).numpy()
    name = "sin_float32_multi"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 tensor with large values
    x = tf.constant(np.array([1000.0, -1000.0], dtype=np.float32)).numpy()
    name = "sin_large_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 tensor with very small values
    x = tf.constant(np.array([1e-6, -1e-6], dtype=np.float32)).numpy()
    name = "sin_small_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 tensor with mixed values
    x = tf.constant(np.array([-np.pi, 0.0, np.pi/4, np.pi/2], dtype=np.float32)).numpy()
    name = "sin_mixed_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.sin"] = tf_math_sin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.sin'.")

check_valid('tf.math.sin', generated_inputs['tf.math.sin'], lib="tf", suffix=0)
