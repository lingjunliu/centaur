
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_norm_inputs():
    """
    Generates a list of valid inputs for the tf.norm function.
    """
    list_of_inputs = []

    # Input 1: Basic Frobenius norm on a 2D float32 tensor
    input_dict_1 = {
        'tensor': np.arange(6, dtype=np.float32).reshape(2, 3),
        'ord': 'fro',
        'axis': (0, 1),
        'keepdims': False,
        'name': 'fro_norm_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Euclidean norm on a batch of matrices (3D tensor) with keepdims
    input_dict_2 = {
        'tensor': np.random.rand(2, 3, 4).astype(np.float32),
        'ord': 'euclidean',
        'axis': (1, 2),
        'keepdims': True,
        'name': 'euclidean_norm_batch_keepdims'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Frobenius norm on a float64 tensor with negative values
    # Changed from ord='1' to 'fro' to resolve ambiguity with the 'string' type constraint.
    input_dict_3 = {
        'tensor': np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float64),
        'ord': 'fro',
        'axis': (0, 1),
        'keepdims': False,
        'name': 'matrix_fro_norm_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Euclidean norm on a 3D tensor
    # Changed from ord='inf' to 'euclidean'
    input_dict_4 = {
        'tensor': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'ord': 'euclidean',
        'axis': (1, 2),
        'keepdims': False,
        'name': 'matrix_euclidean_norm_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Frobenius norm on a 2D tensor
    # Changed from ord='2' to 'fro'
    input_dict_5 = {
        'tensor': np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        'ord': 'fro',
        'axis': (0, 1),
        'keepdims': True,
        'name': 'matrix_fro_norm_2d_keepdims'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Frobenius norm on a 4D tensor, with negative axis indices
    input_dict_6 = {
        'tensor': np.random.rand(2, 3, 4, 5).astype(np.float32),
        'ord': 'fro',
        'axis': (-2, -1),
        'keepdims': True,
        'name': 'fro_norm_4d_neg_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Euclidean norm on a complex128 tensor
    input_dict_7 = {
        'tensor': (np.arange(6, dtype=np.float64).reshape(2, 3) + 1j * np.arange(6, 0, -1, dtype=np.float64).reshape(2, 3)).astype(np.complex128),
        'ord': 'euclidean',
        'axis': (0, 1),
        'keepdims': True,
        'name': 'complex128_euclidean_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Frobenius norm on a complex64 tensor (batch of matrices)
    input_dict_8 = {
        'tensor': (np.random.rand(3, 2, 4) + 1j * np.random.rand(3, 2, 4)).astype(np.complex64),
        'ord': 'fro',
        'axis': (1, 2),
        'keepdims': False,
        'name': 'complex64_fro_norm_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Euclidean norm on a high-rank tensor with non-adjacent axes
    input_dict_9 = {
        'tensor': np.random.rand(2, 3, 4, 5).astype(np.float64),
        'ord': 'euclidean',
        'axis': (1, 3),
        'keepdims': True,
        'name': 'high_rank_non_adj_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Frobenius norm on 3D tensor with mixed positive/negative axis
    # Changed ord from '1' to 'fro'
    input_dict_10 = {
        'tensor': np.arange(-12, 12, dtype=np.float32).reshape(2, 3, 4),
        'ord': 'fro',
        'axis': (0, -1),
        'keepdims': False,
        'name': 'mixed_sign_axis_fro'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Euclidean norm on float64 tensor
    input_dict_11 = {
        'tensor': np.random.rand(4, 2, 3).astype(np.float64),
        'ord': 'euclidean',
        'axis': (0, 2),
        'keepdims': False,
        'name': 'euclidean_norm_float64_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.norm"] = get_tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm'.")

check_valid('tf.norm', generated_inputs['tf.norm'], lib="tf", suffix=0)
