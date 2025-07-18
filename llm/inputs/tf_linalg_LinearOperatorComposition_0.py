
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_tf_linalg_linearoperatorcomposition_inputs():
    """
    Generates a list of valid inputs for tf.linalg.LinearOperatorComposition.

    NOTE: The user's testing framework raises a `TypeError` when the 'operators'
    parameter is a list of `tf.linalg.LinearOperator` objects, as it cannot
    process lists of complex objects. To resolve this specific error, this
    function provides a list of NumPy arrays for the 'operators' parameter,
    adhering to the "inputs should be in numpy format" requirement. It is
    assumed that the execution harness is responsible for converting each
-   NumPy array in the list into a `tf.linalg.LinearOperatorFullMatrix`
    instance before calling the target API.
    """
    list_of_inputs = []

    # Input 1: Simple composition of two 2x2 identity matrices.
    mat1_1 = np.eye(2, dtype=np.float32)
    mat1_2 = np.eye(2, dtype=np.float32)
    input_dict_1 = {
        'operators': [mat1_1, mat1_2],
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'simple_2x2_composition'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Composition of two 3x3 random matrices, float64.
    mat2_1 = np.random.rand(3, 3).astype(np.float64)
    mat2_2 = np.random.rand(3, 3).astype(np.float64)
    input_dict_2 = {
        'operators': [mat2_1, mat2_2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'random_3x3_composition'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Non-square composition (2x3 @ 3x4 -> 2x4).
    mat3_1 = np.random.rand(2, 3).astype(np.float32)
    mat3_2 = np.random.rand(3, 4).astype(np.float32)
    input_dict_3 = {
        'operators': [mat3_1, mat3_2],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'non_square_composition'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A longer chain of compositions (4x4).
    mat4_1 = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    mat4_2 = np.array([[5, 6, 0, 0], [7, 8, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    mat4_3 = np.diag([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    input_dict_4 = {
        'operators': [mat4_1, mat4_2, mat4_3],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'long_chain_4x4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Batch composition (batch_shape [2]).
    mat5_1 = np.random.rand(2, 3, 4).astype(np.float32)
    mat5_2 = np.random.rand(2, 4, 5).astype(np.float32)
    input_dict_5 = {
        'operators': [mat5_1, mat5_2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'batch_composition'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single operator in the list.
    mat6_1 = np.random.rand(5, 5).astype(np.float32)
    input_dict_6 = {
        'operators': [mat6_1],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'single_operator_in_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Broadcasted batch composition.
    mat7_1 = np.random.rand(3, 1, 2, 3).astype(np.float32)
    mat7_2 = np.random.rand(1, 4, 3, 2).astype(np.float32)
    input_dict_7 = {
        'operators': [mat7_1, mat7_2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'broadcast_batch_composition'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Composition resulting in a singular matrix.
    mat8_1 = np.array([[1., 1.], [1., 1.]], dtype=np.float32)
    mat8_2 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    input_dict_8 = {
        'operators': [mat8_1, mat8_2],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'singular_composition'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: High-dimensional batch composition
    mat9_1 = np.random.rand(2, 1, 3, 5, 4).astype(np.float32)
    mat9_2 = np.random.rand(1, 4, 3, 4, 2).astype(np.float32)
    input_dict_9 = {
        'operators': [mat9_1, mat9_2],
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': False,
        'name': 'high_dim_batch_composition'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Composition of four non-square operators
    mat10_1 = np.random.rand(7, 6).astype(np.float32)
    mat10_2 = np.random.rand(6, 5).astype(np.float32)
    mat10_3 = np.random.rand(5, 4).astype(np.float32)
    mat10_4 = np.random.rand(4, 3).astype(np.float32)
    input_dict_10 = {
        'operators': [mat10_1, mat10_2, mat10_3, mat10_4],
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'name': 'long_nonsquare_chain'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorComposition"] = generate_tf_linalg_linearoperatorcomposition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorComposition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorComposition'.")

check_valid('tf.linalg.LinearOperatorComposition', generated_inputs['tf.linalg.LinearOperatorComposition'], lib="tf", suffix=0)
