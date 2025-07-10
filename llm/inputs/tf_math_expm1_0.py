
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_expm1_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = np.float32(2.0)
    name = "scalar_float32"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 scalar
    x = np.float64(2.0)
    name = "scalar_float64"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 array
    x = np.array([2.0, 8.0], dtype=np.float32)
    name = "array_float32"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 array
    x = np.array([2.0, 8.0], dtype=np.float64)
    name = "array_float64"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 scalar
    x = np.complex64(1 + 1j)
    name = "scalar_complex64"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 scalar
    x = np.complex128(1 + 1j)
    name = "scalar_complex128"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    name = "array_bfloat16"
    input_dict = {"x": tf.constant(x, dtype=tf.bfloat16).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    name = "array_half"
    input_dict = {"x": tf.constant(x, dtype=tf.half).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.expm1"] = tf_math_expm1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.expm1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.expm1'.")

check_valid('tf.math.expm1', generated_inputs['tf.math.expm1'], lib="tf", suffix=0)
