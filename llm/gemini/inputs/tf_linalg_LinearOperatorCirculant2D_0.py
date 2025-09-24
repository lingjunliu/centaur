
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorcirculant2d_inputs():
    list_of_inputs = []

    # Input 1
    spectrum = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = True
    is_square = True
    name = "circulant_op_1"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    spectrum = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex64)
    input_output_dtype = tf.float32
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "circulant_op_2"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    spectrum = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex64)
    input_output_dtype = tf.complex128
    is_non_singular = None
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "circulant_op_3"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    spectrum = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "circulant_op_4"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    spectrum = np.array([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "circulant_op_5"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    spectrum = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.complex64)
    input_output_dtype = tf.float64
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "circulant_op_6"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    spectrum = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circulant_op_7"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    spectrum = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = None
    is_square = True
    name = "circulant_op_8"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    spectrum = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], dtype=np.complex64)
    input_output_dtype = tf.float32
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "circulant_op_9"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    spectrum = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    input_output_dtype = tf.complex128
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "circulant_op_10"
    input_dict = {"spectrum": tf.convert_to_tensor(spectrum).numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorCirculant2D"] = tf_linalg_linearoperatorcirculant2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorCirculant2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorCirculant2D'.")

check_valid('tf.linalg.LinearOperatorCirculant2D', generated_inputs['tf.linalg.LinearOperatorCirculant2D'], lib="tf", suffix=0)
