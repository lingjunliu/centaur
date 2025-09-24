
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_polyval_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float coefficients and scalar x
    coeffs = [np.float32(1.0), np.float32(2.5), np.float32(-4.2)]
    x = np.float32(5.0)
    name = None
    input_dict = {"coeffs": coeffs, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer coefficients and scalar x
    coeffs = [np.int32(2), np.int32(1), np.int32(0)]
    x = np.int32(3)
    name = "polynomial_evaluation"
    input_dict = {"coeffs": coeffs, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor x
    coeffs = [np.int32(1), np.int32(2), np.int32(3)]
    x = np.array([1, 2, 3], dtype=np.int32)
    name = None
    input_dict = {"coeffs": coeffs, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher order polynomial with negative coefficients and x values
    coeffs = [np.float32(1.0), np.float32(-2.0), np.float32(3.0), np.float32(-4.0)]
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    name = None
    input_dict = {"coeffs": coeffs, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Tensor x
    coeffs = [np.int32(1), np.int32(2)]
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    name = None
    input_dict = {"coeffs": coeffs, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different shape of coefficients list and x
    coeffs = [np.int32(2), np.int32(1), np.int32(0), np.int32(-1)]
    x = np.array(2, dtype=np.int32)
    name = None
    input_dict = {"coeffs": coeffs, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7:  float64 - REMOVE because of error
    # coeffs = [1.0, 2.5, -4.2]
    # x = np.float64(5.0)
    # name = None
    # input_dict = {"coeffs": coeffs, "x": x, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64 coefficients - REMOVED because of error
    # coeffs = [1+1j, 2.5, -4.2]
    # x = np.float32(5.0)
    # name = None
    # input_dict = {"coeffs": coeffs, "x": x, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: coeffs is empty - REMOVED because of error
    # coeffs = []
    # x = np.float32(5.0)
    # name = None
    # input_dict = {"coeffs": coeffs, "x": x, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x as an array with complex value - REMOVED because of error
    # coeffs = [1.0, 2.5, -4.2]
    # x = np.array([1+1j, 2.0], dtype=np.complex64)
    # name = None
    # input_dict = {"coeffs": coeffs, "x": x, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.polyval"] = tf_math_polyval_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.polyval' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.polyval'.")

check_valid('tf.math.polyval', generated_inputs['tf.math.polyval'], lib="tf", suffix=0)
