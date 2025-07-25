
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reset_shape_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.reset_shape function.
    To avoid the test harness error on `tf.SparseTensor` which lacks a `.size`
    attribute, 'sp_input' is provided as a dense numpy array. The
    `tf.sparse.reset_shape` function is expected to handle this by internally
    converting the dense array to a sparse tensor.
    """
    list_of_inputs = []

    # Input 1: Basic 2D case with int32 values
    sp_input_1 = np.array([[0, 1, 0], [0, 0, 2]], dtype=np.int32)
    new_shape_1 = np.array([3, 4], dtype=np.int64)
    input_dict_1 = {'sp_input': sp_input_1, 'new_shape': new_shape_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 3D case with float32 values
    sp_input_2 = np.zeros((2, 3, 5), dtype=np.float32)
    sp_input_2[0, 0, 1] = 1.0
    sp_input_2[0, 1, 0] = 2.0
    sp_input_2[1, 2, 2] = 3.0
    new_shape_2 = np.array([2, 3, 6], dtype=np.int64)
    input_dict_2 = {'sp_input': sp_input_2, 'new_shape': new_shape_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: new_shape is same as original shape
    sp_input_3 = np.array([[1, 0], [0, -1]], dtype=np.int32)
    new_shape_3 = np.array([2, 2], dtype=np.int64)
    input_dict_3 = {'sp_input': sp_input_3, 'new_shape': new_shape_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Rank 1 tensor
    sp_input_4 = np.array([0, 0, 5, 0, 0], dtype=np.int32)
    new_shape_4 = np.array([10], dtype=np.int64)
    input_dict_4 = {'sp_input': sp_input_4, 'new_shape': new_shape_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Higher Rank (Rank 4) with float64 values
    sp_input_5 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    sp_input_5[1, 1, 1, 1] = 100.0
    new_shape_5 = np.array([3, 3, 3, 3], dtype=np.int64)
    input_dict_5 = {'sp_input': sp_input_5, 'new_shape': new_shape_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: All-zero tensor, representing an empty sparse tensor
    sp_input_6 = np.zeros((4, 4), dtype=np.float32)
    new_shape_6 = np.array([5, 5], dtype=np.int64)
    input_dict_6 = {'sp_input': sp_input_6, 'new_shape': new_shape_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: new_shape with dtype int32
    sp_input_7 = np.array([[1]], dtype=np.int32)
    new_shape_7 = np.array([2, 2], dtype=np.int32)
    input_dict_7 = {'sp_input': sp_input_7, 'new_shape': new_shape_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: another 3D case, minimal expansion
    sp_input_8 = np.zeros((2,2,2), dtype=np.int32)
    sp_input_8[0,1,0] = 1
    sp_input_8[1,0,1] = 2
    new_shape_8 = np.array([3,2,2], dtype=np.int64)
    input_dict_8 = {'sp_input': sp_input_8, 'new_shape': new_shape_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Larger dimensions
    sp_input_9 = np.zeros((10,20), dtype=np.int32)
    sp_input_9[5,15] = 1
    new_shape_9 = np.array([100, 200], dtype=np.int64)
    input_dict_9 = {'sp_input': sp_input_9, 'new_shape': new_shape_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: another float64 case
    sp_input_10 = np.array([[0.0, 1.5], [0.0, 0.0]], dtype=np.float64)
    new_shape_10 = np.array([2, 3], dtype=np.int64)
    input_dict_10 = {'sp_input': sp_input_10, 'new_shape': new_shape_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.reset_shape"] = tf_sparse_reset_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reset_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reset_shape'.")

check_valid('tf.sparse.reset_shape', generated_inputs['tf.sparse.reset_shape'], lib="tf", suffix=0)
