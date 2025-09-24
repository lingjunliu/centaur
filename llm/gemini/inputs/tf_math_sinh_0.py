
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sinh_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = "sinh_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D
    x = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    name = "sinh_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, 1D with inf.  Numpy does not directly support bfloat16.
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    name = "sinh_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 1D
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float16)
    name = "sinh_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 1D
    x = np.array([1+1j, 2-2j, 3+0j, 0-4j], dtype=np.complex64)
    name = "sinh_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 2D
    x = np.array([[1+1j, 2-2j], [3+0j, 0-4j]], dtype=np.complex128)
    name = "sinh_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D
    x = np.random.rand(2, 3, 4).astype(np.float32)
    name = "sinh_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, scalar
    x = np.array(1.5, dtype=np.float64)
    name = "sinh_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, large values
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    name = "sinh_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, small values
    x = np.array([-0.0001, 0.0, 0.0001], dtype=np.float64)
    name = "sinh_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.sinh"] = tf_math_sinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.sinh'.")

check_valid('tf.math.sinh', generated_inputs['tf.math.sinh'], lib="tf", suffix=0)
