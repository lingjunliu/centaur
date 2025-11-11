
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_linear_operator_lower_triangular_inputs():
    list_of_inputs = []

    
    # Input 1: Basic 2x2 lower triangular matrix
    tril = np.array([[1., 0.], [2., 3.]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": True,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: 3x3 lower triangular matrix with non-singular flag
    tril = np.array([[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: 4x4 lower triangular matrix with negative values
    tril = np.array([[-1., 0., 0., 0.], [2., -3., 0., 0.], [4., 5., -6., 0.], [7., 8., 9., -10.]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: 2x2 batch of lower triangular matrices
    tril = np.array([[[1., 0.], [2., 3.]], [[4., 0.], [5., 6.]]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": True,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: 3x3 batch of lower triangular matrices with different values
    tril = np.array([[[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]], [[7., 0., 0.], [8., 9., 0.], [10., 11., 12.]]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: 5x5 lower triangular matrix with zero values
    tril = np.array([[1., 0., 0., 0., 0.], [2., 3., 0., 0., 0.], [4., 5., 6., 0., 0.], [7., 8., 9., 10., 0.], [11., 12., 13., 14., 15.]])
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: 3x3 batch of 4x4 lower triangular matrices
    tril = np.array([[[[1., 0., 0., 0.], [2., 3., 0., 0.], [4., 5., 6., 0.], [7., 8., 9., 10.]], [[11., 0., 0., 0.], [12., 13., 0., 0.], [14., 15., 16., 0.], [17., 18., 19., 20.]]]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: 4x4 batch of 3x3 lower triangular matrices with different dimensions
    tril = np.array([[[[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]], [[7., 0., 0.], [8., 9., 0.], [10., 11., 12.]]], [[13., 0., 0.], [14., 15., 0.], [16., 17., 18.]]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: 4x4 lower triangular matrix with floating point values
    tril = np.array([[1.5, 0., 0., 0.], [2.7, 3.1, 0., 0.], [4.8, 5.9, 6.2, 0.], [7.1, 8.3, 9.4, 10.5]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": True,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: 2x2 batch of 3x3 lower triangular matrices with different shapes
    tril = np.array([[[[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]], [[7., 0., 0.], [8., 9., 0.], [10., 11., 12.]]], [[13., 0., 0.], [14., 15., 0.], [16., 17., 18.]]])
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "LinearOperatorLowerTriangular"
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorLowerTriangular"] = generate_linear_operator_lower_triangular_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorLowerTriangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorLowerTriangular'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorLowerTriangular', generated_inputs['tf.linalg.LinearOperatorLowerTriangular'], lib="tf", suffix=0)
