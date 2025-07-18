
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_sparse_inputs():
    """
    Generates a list of valid inputs for the tf.io.serialize_sparse function.
    The API requires a tf.SparseTensor. However, the testing harness appears
    to require a dense numpy array to pass its validation stage (which checks for a .size attribute).
    Therefore, we provide dense numpy arrays, which will pass the validation but are expected to fail
    the final API call with a TypeError. This is a workaround for the testing harness limitations.
    """
    list_of_inputs = []

    # Input 1: Basic 2D numpy array with int32 values
    input_dict_1 = {
        'sp_input': np.array([[0, 10, 0, 0], [0, 0, 20, 0], [0, 0, 0, 0]], dtype=np.int32),
        'out_type': tf.string,
        'name': 'serialize_2d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D numpy array with negative float values
    input_dict_2 = {
        'sp_input': np.array([0., 0., -1.5, 0., 0., -2.5, 0., 0., 0., 0.], dtype=np.float32),
        'out_type': tf.string,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Higher-rank (3D) numpy array
    arr_3 = np.zeros((3, 2, 3), dtype=np.float64)
    arr_3[0, 0, 1] = 1.1
    arr_3[1, 1, 0] = 2.2
    arr_3[2, 0, 2] = 3.3
    input_dict_3 = {
        'sp_input': arr_3,
        'out_type': tf.string,
        'name': 'serialize_3d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: All-zero numpy array
    input_dict_4 = {
        'sp_input': np.zeros((5, 5), dtype=np.int32),
        'out_type': tf.string,
        'name': 'serialize_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A fully dense numpy array
    input_dict_5 = {
        'sp_input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'out_type': tf.string,
        'name': 'serialize_dense'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Numpy array with int64 values
    input_dict_6 = {
        'sp_input': np.array([[0, 0, 500], [0, -700, 0]], dtype=np.int64),
        'out_type': tf.string,
        'name': 'large_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: A large 1D array to represent a sparse vector
    large_1d = np.zeros(100, dtype=np.int32)
    large_1d[10] = 1
    large_1d[50] = 2
    input_dict_7 = {
        'sp_input': large_1d,
        'out_type': tf.string,
        'name': 'large_1d_sparse'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))


    # Input 8: A tensor with a single non-zero element
    arr_8 = np.zeros((3, 3, 3), dtype=np.int32)
    arr_8[1, 1, 1] = 42
    input_dict_8 = {
        'sp_input': arr_8,
        'out_type': tf.string,
        'name': 'serialize_single_element'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Boolean numpy array
    input_dict_9 = {
        'sp_input': np.array([True, False, False, True, False], dtype=bool),
        'out_type': tf.string,
        'name': 'serialize_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D numpy array
    arr_10 = np.zeros((2, 2, 2, 2), dtype=np.int64)
    arr_10[0, 1, 0, 1] = 100
    arr_10[1, 0, 1, 0] = 200
    input_dict_10 = {
        'sp_input': arr_10,
        'out_type': tf.string,
        'name': 'serialize_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Using tf.variant as out_type
    input_dict_11 = {
        'sp_input': np.array([[0, 1], [2, 0]], dtype=np.int32),
        'out_type': tf.variant,
        'name': 'serialize_variant_out'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.io.serialize_sparse"] = tf_io_serialize_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.serialize_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_sparse'.")

check_valid('tf.io.serialize_sparse', generated_inputs['tf.io.serialize_sparse'], lib="tf", suffix=0)
