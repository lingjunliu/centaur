
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_tensordot_inputs():
    """
    Generates a list of valid inputs for the tf.tensordot function.
    """
    list_of_inputs = []

    # Input 1: Matrix multiplication (axes=1)
    input_dict_1 = {
        'a': np.arange(6, dtype=np.float32).reshape(2, 3),
        'b': np.arange(12, dtype=np.float32).reshape(3, 4),
        'axes': 1,
        'name': 'matrix_multiplication'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Vector dot product (axes=1)
    input_dict_2 = {
        'a': np.array([1, 2, 3], dtype=np.float64),
        'b': np.array([4, 5, 6], dtype=np.float64),
        'axes': 1,
        'name': 'vector_dot_product'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Outer product (axes=0)
    input_dict_3 = {
        'a': np.array([1, 2], dtype=np.float32),
        'b': np.array([3, 4, 5], dtype=np.float32),
        'axes': 0,
        'name': 'vector_outer_product'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Outer product of two matrices (axes=0)
    input_dict_4 = {
        'a': np.arange(6, dtype=np.int32).reshape(2, 3),
        'b': np.arange(4, dtype=np.int32).reshape(2, 2),
        'axes': 0,
        'name': 'matrix_outer_product'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Contraction of 3D tensors (axes=2)
    input_dict_5 = {
        'a': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'b': np.arange(60, dtype=np.float32).reshape(3, 4, 5),
        'axes': 2,
        'name': '3d_tensor_contraction_axes2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Contraction of 3D and 2D tensors (axes=1)
    input_dict_6 = {
        'a': np.arange(24, dtype=np.float64).reshape(4, 3, 2),
        'b': np.arange(10, dtype=np.float64).reshape(2, 5),
        'axes': 1,
        'name': '3d_2d_contraction_axes1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using float64 and negative values
    input_dict_7 = {
        'a': np.arange(-6, 0, dtype=np.float64).reshape(3, 2),
        'b': np.arange(8, dtype=np.float64).reshape(2, 4),
        'axes': 1,
        'name': 'float64_with_negatives'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Batch matrix multiplication style (axes=2)
    input_dict_8 = {
        'a': np.random.randn(10, 2, 3).astype(np.float32),
        'b': np.random.randn(2, 3, 5).astype(np.float32),
        'axes': 2,
        'name': 'batch_matrix_multiplication_style'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Higher rank contraction (axes=3)
    input_dict_9 = {
        'a': np.random.uniform(size=(2, 3, 4, 5)).astype(np.float32),
        'b': np.random.uniform(size=(3, 4, 5, 6)).astype(np.float32),
        'axes': 3,
        'name': '4d_tensor_contraction'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Contraction involving a dimension of size 1
    input_dict_10 = {
        'a': np.random.randn(5, 1, 6).astype(np.float32),
        'b': np.random.randn(1, 6, 4).astype(np.float32),
        'axes': 2,
        'name': 'contraction_with_dim_one'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Contraction where a's rank equals axes
    input_dict_11 = {
        'a': np.arange(6, dtype=np.int32).reshape(2, 3),
        'b': np.arange(18, dtype=np.int32).reshape(2, 3, 3),
        'axes': 2,
        'name': 'rank_equals_axes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Fixed 2D and 4D contraction
    input_dict_12 = {
        'a': np.arange(12, dtype=np.float32).reshape(3, 4),
        'b': np.arange(48, dtype=np.float32).reshape(4, 2, 3, 2),
        'axes': 1,
        'name': '2d_4d_contraction'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.tensordot"] = tf_tensordot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.tensordot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tensordot'.")

check_valid('tf.tensordot', generated_inputs['tf.tensordot'], lib="tf", suffix=0)
