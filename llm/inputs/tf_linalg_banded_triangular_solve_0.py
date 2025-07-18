
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_linalg_banded_triangular_solve_inputs():
    """
    This function generates a list of valid inputs for the
    tf.linalg.banded_triangular_solve function.
    """
    list_of_inputs = []

    # Input 1: Basic lower triangular, 3x3, 2 bands, float32
    # Matrix A = [[2, 0, 0], [1, 3, 0], [0, 2, 4]]
    # rhs = [[2], [4], [6]]. Solution is [1, 1, 1]
    bands1 = np.array([[2., 3., 4.], [0., 1., 2.]], dtype=np.float32)
    rhs1 = np.array([[2.], [4.], [6.]], dtype=np.float32)
    input_dict_1 = {
        'bands': bands1,
        'rhs': rhs1,
        'lower': True,
        'adjoint': False,
        'name': 'basic_lower_3x3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic upper triangular, 4x4, 2 bands, float32
    # Matrix A = [[5, 1, 0, 0], [0, 4, 1, 0], [0, 0, 3, 1], [0, 0, 0, 2]]
    # rhs = [[6], [5], [4], [2]]. Solution is [1, 1, 1, 1]
    bands2 = np.array([[1., 1., 1., 0.], [5., 4., 3., 2.]], dtype=np.float32)
    rhs2 = np.array([[6.], [5.], [4.], [2.]], dtype=np.float32)
    input_dict_2 = {
        'bands': bands2,
        'rhs': rhs2,
        'lower': False,
        'adjoint': False,
        'name': 'basic_upper_4x4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Adjoint on lower triangular (solves A.T * x = rhs), float32
    # A.T = [[2, 1, 0], [0, 3, 2], [0, 0, 4]]
    # rhs = [[3], [5], [4]]. Solution is [1, 1, 1]
    bands3 = np.array([[2., 3., 4.], [0., 1., 2.]], dtype=np.float32)
    rhs3 = np.array([[3.], [5.], [4.]], dtype=np.float32)
    input_dict_3 = {
        'bands': bands3,
        'rhs': rhs3,
        'lower': True,
        'adjoint': True,
        'name': 'adjoint_lower_3x3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Adjoint on upper triangular (solves A.T * x = rhs), float32
    # A.T = [[5, 0, 0, 0], [1, 4, 0, 0], [0, 1, 3, 0], [0, 0, 1, 2]]
    # rhs = [[5], [5], [4], [3]]. Solution is [1, 1, 1, 1]
    bands4 = np.array([[1., 1., 1., 0.], [5., 4., 3., 2.]], dtype=np.float32)
    rhs4 = np.array([[5.], [5.], [4.], [3.]], dtype=np.float32)
    input_dict_4 = {
        'bands': bands4,
        'rhs': rhs4,
        'lower': False,
        'adjoint': True,
        'name': 'adjoint_upper_4x4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Batched input for bands and rhs, float64
    bands5 = np.array([
        [[2., 3., 4.], [0., 1., 2.]],  # Matrix A1 from Input 1
        [[1., 1., 1.], [0., 2., 3.]]   # Matrix A2 = [[1,0,0],[2,1,0],[0,3,1]]
    ], dtype=np.float64)
    rhs5 = np.array([
        [[2.], [4.], [6.]],  # rhs for A1, soln [1,1,1]
        [[1.], [3.], [4.]]   # rhs for A2, soln [1,1,1]
    ], dtype=np.float64)
    input_dict_5 = {
        'bands': bands5,
        'rhs': rhs5,
        'lower': True,
        'adjoint': False,
        'name': 'batched_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Broadcasting rhs to batched bands
    rhs6 = np.array([[1.], [1.], [1.]], dtype=np.float64)
    input_dict_6 = {
        'bands': bands5, # from input 5
        'rhs': rhs6,
        'lower': True,
        'adjoint': False,
        'name': 'broadcast_rhs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: rhs is a matrix (multiple right-hand sides)
    # A = [[2, 0, 0], [1, 3, 0], [0, 2, 4]]
    # rhs1 = [2,4,6], soln1 = [1,1,1]
    # rhs2 = [4,7,10], soln2 = [2, 5/3, 5/3]
    bands7 = np.array([[2., 3., 4.], [0., 1., 2.]], dtype=np.float32)
    rhs7 = np.array([[2., 4.], [4., 7.], [6., 10.]], dtype=np.float32)
    input_dict_7 = {
        'bands': bands7,
        'rhs': rhs7,
        'lower': True,
        'adjoint': False,
        'name': 'multiple_rhs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex numbers, lower triangular, complex64
    # A = [[2j, 0], [1, 3j]]. rhs = [[2j], [1+3j]]. Soln [1, 1]
    bands8 = np.array([[0.+2.j, 0.+3.j], [0., 1.+0.j]], dtype=np.complex64)
    rhs8 = np.array([[0.+2.j], [1.+3.j]], dtype=np.complex64)
    input_dict_8 = {
        'bands': bands8,
        'rhs': rhs8,
        'lower': True,
        'adjoint': False,
        'name': 'complex64_lower'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Complex numbers with adjoint, complex128
    # A = [[2j, 0], [1, 3j]]. A_H = [[-2j, 1], [0, -3j]]
    # rhs = [[-2j+1], [-3j]]. Soln [1, 1]
    bands9 = np.array([[0.+2.j, 0.+3.j], [0., 1.+0.j]], dtype=np.complex128)
    rhs9 = np.array([[-2.j+1.], [-3.j]], dtype=np.complex128)
    input_dict_9 = {
        'bands': bands9,
        'rhs': rhs9,
        'lower': True,
        'adjoint': True,
        'name': 'complex128_adjoint'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Lower triangular with more bands (K=3, M=4)
    # A = [[1,0,0,0], [2,1,0,0], [3,2,1,0], [0,3,2,1]]
    # rhs = [[1], [3], [6], [6]]. Soln [1,1,1,1]
    bands10 = np.array([[1., 1., 1., 1.], [0., 2., 2., 2.], [0., 0., 3., 3.]], dtype=np.float32)
    rhs10 = np.array([[1.], [3.], [6.], [6.]], dtype=np.float32)
    input_dict_10 = {
        'bands': bands10,
        'rhs': rhs10,
        'lower': True,
        'adjoint': False,
        'name': 'lower_3_bands'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Upper triangular with more bands (K=3, M=4), lower=False
    # A = [[1,2,3,0], [0,1,2,3], [0,0,1,2], [0,0,0,1]]
    # rhs = [[6], [6], [3], [1]]. Soln [1,1,1,1]
    bands11 = np.array([[3., 3., 0., 0.], [2., 2., 2., 0.], [1., 1., 1., 1.]], dtype=np.float32)
    rhs11 = np.array([[6.], [6], [3.], [1.]], dtype=np.float32)
    input_dict_11 = {
        'bands': bands11,
        'rhs': rhs11,
        'lower': False,
        'adjoint': False,
        'name': 'upper_3_bands'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Minimal case, 1x1 matrix
    bands12 = np.array([[-5.]], dtype=np.float32)
    rhs12 = np.array([[10.]], dtype=np.float32)
    input_dict_12 = {
        'bands': bands12,
        'rhs': rhs12,
        'lower': True,
        'adjoint': False,
        'name': 'minimal_1x1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.banded_triangular_solve"] = tf_linalg_banded_triangular_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.banded_triangular_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.banded_triangular_solve'.")

check_valid('tf.linalg.banded_triangular_solve', generated_inputs['tf.linalg.banded_triangular_solve'], lib="tf", suffix=0)
