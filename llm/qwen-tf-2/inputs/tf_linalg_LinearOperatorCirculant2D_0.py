
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_linear_operator_circulant2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic 2D spectrum with complex dtype
    spectrum = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - 3D spectrum with real dtype
    spectrum = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 4D spectrum with complex dtype and non-singular flag
    spectrum = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 2D spectrum with real dtype and self-adjoint flag
    spectrum = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - 2D spectrum with complex dtype and positive definite flag
    spectrum = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = True
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - 3D spectrum with real dtype and square flag set to True (this is valid)
    spectrum = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - 2D spectrum with negative values and complex dtype
    spectrum = np.array([[-1., 2., 3.], [4., -5., 6.], [7., 8., -9.]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - 4D spectrum with complex dtype and different dimensions
    spectrum = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - 3D spectrum with real dtype and different batch dimensions
    spectrum = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - 2D spectrum with complex dtype and different name
    spectrum = np.array([[1., 2.], [3., 4.]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorCirculant2D"] = generate_linear_operator_circulant2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorCirculant2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorCirculant2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorCirculant2D', generated_inputs['tf.linalg.LinearOperatorCirculant2D'], lib="tf", suffix=0)
