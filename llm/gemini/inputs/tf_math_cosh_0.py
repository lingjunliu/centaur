
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_cosh_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    x = np.float32(0.0)
    name = "cosh_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 1D array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    name = "cosh_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 2D array
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    name = "cosh_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 scalar
    x = np.float64(0.5)
    name = "cosh_scalar_double"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 1D array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    name = "cosh_1d_double"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 2D array
    x = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    name = "cosh_2d_double"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 scalar
    x = np.array(1.0, dtype=np.float16)
    name = "cosh_bfloat16_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half scalar
    x = np.array(1.0, dtype=np.float16)
    name = "cosh_half_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64 1D array
    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    name = "cosh_complex64_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: complex128 2D array
    x = np.array([[1+1j, 2-2j], [3+0j, 0-1j]], dtype=np.complex128)
    name = "cosh_complex128_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float32 scalar negative
    x = np.float32(-5.0)
    name = "cosh_scalar_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.cosh"] = tf_math_cosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cosh'.")

check_valid('tf.math.cosh', generated_inputs['tf.math.cosh'], lib="tf", suffix=0)
