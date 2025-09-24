
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_sparse_inputs():
    list_of_inputs = []

    # The 'sp_input' is provided as a dense numpy array to conform to the
    # testing framework's expectation for the 'tensor' type, which requires
    # a .size attribute. The framework is expected to convert this dense
    # array to a tf.SparseTensor before calling the API.

    # Input 1: Basic 2D int32 tensor
    sp_input_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]], dtype=np.int32)
    input_dict_1 = {
        'sp_input': sp_input_1,
        'out_type': np.string_,
        'name': 'basic_2d_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float32 tensor with negative values
    sp_input_2 = np.array([[-1.5, 0.0], [0.0, 2.5]], dtype=np.float32)
    input_dict_2 = {
        'sp_input': sp_input_2,
        'out_type': np.string_,
        'name': '2d_float32_negative'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D tensor (vector) of float64
    sp_input_3 = np.array([0, 10.0, 0, 20.0, 0, 30.0, 0, 0, 0, 0], dtype=np.float64)
    input_dict_3 = {
        'sp_input': sp_input_3,
        'out_type': np.string_,
        'name': '1d_vector_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D int64 tensor
    sp_input_4 = np.zeros((2, 2, 2), dtype=np.int64)
    sp_input_4[0, 0, 1] = 100
    sp_input_4[1, 1, 0] = 200
    input_dict_4 = {
        'sp_input': sp_input_4,
        'out_type': np.string_,
        'name': '3d_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: All-zero tensor (represents empty sparse tensor)
    sp_input_5 = np.zeros((5, 5), dtype=np.float32)
    input_dict_5 = {
        'sp_input': sp_input_5,
        'out_type': np.string_,
        'name': 'empty_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Fully dense tensor
    sp_input_6 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict_6 = {
        'sp_input': sp_input_6,
        'out_type': np.string_,
        'name': 'full_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High-rank (4D) tensor
    sp_input_7 = np.zeros((2, 3, 4, 5), dtype=np.float32)
    sp_input_7[0, 1, 2, 3] = 1.0
    sp_input_7[1, 2, 3, 4] = -1.0
    input_dict_7 = {
        'sp_input': sp_input_7,
        'out_type': np.string_,
        'name': 'high_rank_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensor with bool values
    sp_input_8 = np.array([[True, False], [False, True]], dtype=bool)
    input_dict_8 = {
        'sp_input': sp_input_8,
        'out_type': np.string_,
        'name': '2d_bool_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensor with complex64 values
    sp_input_9 = np.array([1 + 2j, 0, 3 - 4j], dtype=np.complex64)
    input_dict_9 = {
        'sp_input': sp_input_9,
        'out_type': np.string_,
        'name': '1d_complex_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensor with large shape and name=None
    sp_input_10 = np.zeros((100, 100), dtype=np.int8)
    sp_input_10[10, 20] = 5
    input_dict_10 = {
        'sp_input': sp_input_10,
        'out_type': np.string_,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
