
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_cos_inputs():
    list_of_inputs = []

    # Input 1: float32, single element
    x = np.array(0.0, dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, vector
    x = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, matrix
    x = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    input_dict = {"x": tf.constant(x).numpy(), "name": "cos_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, positive and negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, multi-dimensional
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict = {"x": tf.constant(x).numpy(), "name": "complex_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, large values
    x = np.array([1000.0, -1000.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, 3D tensor
    x = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "name": "3d_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, array with nan and inf
    x = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, tensor with a large number
    x = np.array([1000000.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.cos"] = tf_math_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cos'.")

check_valid('tf.math.cos', generated_inputs['tf.math.cos'], lib="tf", suffix=0)
