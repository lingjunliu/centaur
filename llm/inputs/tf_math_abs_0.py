
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_abs_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D, positive
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D, negative
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "abs_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D, mixed
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, 1D, positive
    x = np.array([1, 2, 3], dtype=np.int32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32, 1D, negative
    x = np.array([-1, -2, -3], dtype=np.int32)
    name = "abs_int_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, 2D, mixed
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, 1D
    x = np.array([1+1j, 2-2j, -3+3j], dtype=np.complex64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128, 2D
    x = np.array([[1+1j, 2-2j], [-3+3j, -4-4j]], dtype=np.complex128)
    name = "abs_complex_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, 3D
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64, 1D, mixed
    x = np.array([-1, 0, 1], dtype=np.int64)
    name = "abs_int64_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.abs"] = tf_math_abs_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.abs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.abs'.")

check_valid('tf.math.abs', generated_inputs['tf.math.abs'], lib="tf", suffix=0)
