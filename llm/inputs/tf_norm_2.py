
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_norm_inputs():
    """
    Generates a list of valid inputs for the tf.norm function.
    """
    list_of_inputs = []

    # Input 1: 2D float32 matrix, Frobenius norm
    input_dict_1 = {
        'tensor': np.array([[1., 2.], [3., 4.]]).astype(np.float32),
        'ord': 'fro',
        'axis': [0, 1],
        'keepdims': True,
        'name': 'frobenius_norm_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float32 matrix, Euclidean norm
    input_dict_2 = {
        'tensor': np.array([[1., -2.], [-3., 4.]]).astype(np.float32),
        'ord': 'euclidean',
        'axis': [0, 1],
        'keepdims': False,
        'name': 'euclidean_norm_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D float32 tensor, batch of matrices, Frobenius norm on last two dims
    input_dict_3 = {
        'tensor': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'ord': 'fro',
        'axis': [-2, -1],
        'keepdims': True,
        'name': 'batch_matrix_fro_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D float32 tensor, batch of matrices, Euclidean norm on axes (0, 2)
    input_dict_4 = {
        'tensor': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'ord': 'euclidean',
        'axis': [0, 2],
        'keepdims': False,
        'name': 'batch_matrix_euclidean_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D complex128 matrix, Frobenius norm
    input_dict_5 = {
        'tensor': np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]).astype(np.complex128),
        'ord': 'fro',
        'axis': [0, 1],
        'keepdims': True,
        'name': 'complex128_matrix_fro_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D float64 matrix, Euclidean norm
    input_dict_6 = {
        'tensor': np.array([[1., 2., 3.], [4., 5., 6.]]).astype(np.float64),
        'ord': 'euclidean',
        'axis': [0, 1],
        'keepdims': False,
        'name': 'matrix_euclidean_norm_f64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 4D tensor, Frobenius norm over axes 1 and 2
    input_dict_7 = {
        'tensor': np.random.rand(2, 3, 4, 5).astype(np.float32),
        'ord': 'fro',
        'axis': [1, 2],
        'keepdims': True,
        'name': '4d_matrix_fro_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 4D tensor, Euclidean norm over axes 0 and 3
    input_dict_8 = {
        'tensor': np.random.rand(2, 3, 4, 5).astype(np.float32),
        'ord': 'euclidean',
        'axis': [0, 3],
        'keepdims': False,
        'name': '4d_matrix_euclidean_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: 2D complex64 matrix, Euclidean norm
    input_dict_9 = {
        'tensor': np.array([[1+2j, -3+4j], [5-6j, -7-8j]]).astype(np.complex64),
        'ord': 'euclidean',
        'axis': [0, 1],
        'keepdims': False,
        'name': 'complex64_matrix_euclidean_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 5D float32 tensor, Frobenius norm over axes 2 and 4
    input_dict_10 = {
        'tensor': np.random.rand(2, 2, 2, 2, 2).astype(np.float32),
        'ord': 'fro',
        'axis': [2, 4],
        'keepdims': True,
        'name': '5d_matrix_fro_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.norm_2"] = tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm_2'.")

check_valid('tf.norm', generated_inputs['tf.norm_2'], lib="tf", suffix=2)
