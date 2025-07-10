
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorCirculant3D_inputs():
    list_of_inputs = []

    # Input 1
    spectrum = tf.constant(np.random.rand(8, 8, 8) + 1j * np.random.rand(8, 8, 8), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = None
    is_square = True
    name = "circulant_op_1"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    spectrum = tf.constant(np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circulant_op_2"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    spectrum = tf.constant(np.random.rand(2, 2, 2) + 1j * np.random.rand(2, 2, 2), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = True
    is_square = True
    name = "circulant_op_3"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4 - Complex128
    spectrum = tf.constant(np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4), dtype=tf.complex128)
    input_output_dtype = tf.complex128
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "circulant_op_4"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Batch dimension
    spectrum = tf.constant(np.random.rand(2, 4, 4, 4) + 1j * np.random.rand(2, 4, 4, 4), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = True
    is_square = True
    name = "circulant_op_5"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - Different Shape
    spectrum = tf.constant(np.random.rand(3, 5, 7) + 1j * np.random.rand(3, 5, 7), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "circulant_op_6"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - More batch dimensions
    spectrum = tf.constant(np.random.rand(2, 3, 4, 2, 2, 2) + 1j * np.random.rand(2, 3, 4, 2, 2, 2), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "circulant_op_7"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Smaller shape
    spectrum = tf.constant(np.random.rand(1, 1, 1) + 1j * np.random.rand(1, 1, 1), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "circulant_op_8"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - larger dimension
    spectrum = tf.constant(np.random.rand(16, 16, 16) + 1j * np.random.rand(16, 16, 16), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = True
    is_square = True
    name = "circulant_op_9"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - is_square = false
    spectrum = tf.constant(np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4), dtype=tf.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = None
    is_square = True
    name = "circulant_op_10"
    input_dict = {"spectrum": spectrum.numpy(), "input_output_dtype": input_output_dtype, "is_non_singular": is_non_singular,
                  "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite,
                  "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorCirculant3D"] = tf_linalg_LinearOperatorCirculant3D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorCirculant3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorCirculant3D'.")

check_valid('tf.linalg.LinearOperatorCirculant3D', generated_inputs['tf.linalg.LinearOperatorCirculant3D'], lib="tf", suffix=0)
