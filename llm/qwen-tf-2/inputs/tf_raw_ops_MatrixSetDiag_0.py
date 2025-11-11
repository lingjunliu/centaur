
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_matrix_set_diag_inputs():
    list_of_inputs = []
    
    # Input 1: 2D matrix with 1 diagonal
    input_2d = np.array([[1, 2], [3, 4]])
    diagonal_1d = np.array([5, 6])
    input_dict = {
        "name": "test1",
        "input": input_2d,
        "diagonal": diagonal_1d
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D matrix with 2 diagonals
    input_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    diagonal_2d = np.array([[9, 10], [11, 12]])
    input_dict = {
        "name": "test2",
        "input": input_3d,
        "diagonal": diagonal_2d
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D matrix with 3 diagonals
    input_4d = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    diagonal_3d = np.array([[[17, 18], [19, 20]], [[21, 22], [23, 24]]])
    input_dict = {
        "name": "test3",
        "input": input_4d,
        "diagonal": diagonal_3d
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 5D matrix with 4 diagonals
    input_5d = np.array([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    diagonal_4d = np.array([[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[25, 26], [27, 28]]])
    input_dict = {
        "name": "test4",
        "input": input_5d,
        "diagonal": diagonal_4d
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D matrix with negative values
    input_3d_neg = np.array([[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]])
    diagonal_3d_neg = np.array([-9, 10])
    input_dict = {
        "name": "test5",
        "input": input_3d_neg,
        "diagonal": diagonal_3d_neg
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D matrix with different shape
    input_2d_diff = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    diagonal_2d_diff = np.array([10, 11, 12])
    input_dict = {
        "name": "test6",
        "input": input_2d_diff,
        "diagonal": diagonal_2d_diff
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D matrix with floating point values
    input_3d_float = np.array([[[1.5, 2.7], [3.1, 4.8]], [[5.2, 6.9], [7.3, 8.4]]])
    diagonal_3d_float = np.array([9.1, 10.2])
    input_dict = {
        "name": "test7",
        "input": input_3d_float,
        "diagonal": diagonal_3d_float
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D matrix with zero values
    input_4d_zero = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]])
    diagonal_4d_zero = np.array([[16, 17], [18, 19]])
    input_dict = {
        "name": "test8",
        "input": input_4d_zero,
        "diagonal": diagonal_4d_zero
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D matrix with large values
    input_3d_large = np.array([[[100, 200], [300, 400]], [[500, 600], [700, 800]]])
    diagonal_3d_large = np.array([900, 1000])
    input_dict = {
        "name": "test9",
        "input": input_3d_large,
        "diagonal": diagonal_3d_large
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D matrix with mixed values
    input_2d_mixed = np.array([[1, -2], [3, -4]])
    diagonal_2d_mixed = np.array([5, 6])
    input_dict = {
        "name": "test10",
        "input": input_2d_mixed,
        "diagonal": diagonal_2d_mixed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixSetDiag"] = generate_matrix_set_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixSetDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MatrixSetDiag', generated_inputs['tf.raw_ops.MatrixSetDiag'], lib="tf", suffix=0)
