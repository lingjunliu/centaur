
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorcirculant_inputs():
    list_of_inputs = []

    # Input 1
    spectrum = np.array([1.0, 2.0, 3.0], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circ1"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    spectrum = np.array([6.0, 4.0, 2.0], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "circ2"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    spectrum = np.array([1, 1j, -1j], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = False
    is_positive_definite = None
    is_square = True
    name = "circ3"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    spectrum = np.array([6.0, 4.0, 2.0, 4.0], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "circ4"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    spectrum = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circ5"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    spectrum = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_output_dtype = tf.complex128
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circ6"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    spectrum = np.array([6.0, 4.0, 2.0], dtype=np.float64)
    input_output_dtype = tf.float64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "circ7"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    spectrum = np.array([1, 1j, -1j], dtype=np.complex128)
    input_output_dtype = tf.complex128
    is_non_singular = None
    is_self_adjoint = False
    is_positive_definite = None
    is_square = True
    name = "circ8"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    spectrum = np.array([6.0, 4.0, 2.0, 4.0], dtype=np.float64)
    input_output_dtype = tf.float64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "circ9"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    spectrum = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    input_output_dtype = tf.complex128
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circ10"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    spectrum = np.array([1., 2., 1.], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circ11"
    input_dict = {"spectrum": spectrum, "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorCirculant"] = tf_linalg_linearoperatorcirculant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorCirculant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorCirculant'.")

check_valid('tf.linalg.LinearOperatorCirculant', generated_inputs['tf.linalg.LinearOperatorCirculant'], lib="tf", suffix=0)
