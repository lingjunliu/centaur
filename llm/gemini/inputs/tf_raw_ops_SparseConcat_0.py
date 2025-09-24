
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_sparsecancat_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.SparseConcat.
    """
    list_of_inputs = []

    # Input 1: Basic 2D concat, dim=1, string values
    input_dict_1 = {
        'name': 'concat_2d_dim1_string',
        'indices': np.array([
            [[0, 2], [1, 1]],
            [[0, 1], [0, 2]]
        ], dtype=np.int64),
        'values': np.array([
            [b'a', b'c'],
            [b'd', b'e']
        ], dtype=object),
        'shapes': np.array([
            [2, 3],
            [2, 4]
        ], dtype=np.int64),
        'concat_dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2D concat, dim=0, int32 values
    input_dict_2 = {
        'name': 'concat_2d_dim0_int32',
        'indices': np.array([
            [[0, 2], [1, 0]],
            [[0, 1], [2, 4]]
        ], dtype=np.int64),
        'values': np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.int32),
        'shapes': np.array([
            [2, 5],
            [3, 5]
        ], dtype=np.int64),
        'concat_dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Float32 values, concat 3 tensors
    input_dict_3 = {
        'name': 'concat_3_tensors_float32',
        'indices': np.array([
            [[1, 1]],
            [[0, 0]],
            [[1, 0]]
        ], dtype=np.int64),
        'values': np.array([
            [1.1],
            [3.3],
            [5.5]
        ], dtype=np.float32),
        'shapes': np.array([
            [4, 2],
            [4, 3],
            [4, 1]
        ], dtype=np.int64),
        'concat_dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Concat tensors with different values
    input_dict_4 = {
        'name': 'concat_with_different_values',
        'indices': np.array([
            [[0, 2], [1, 0]],
            [[0, 0], [1, 1]]
        ], dtype=np.int64),
        'values': np.array([
            [-10, -20],
            [-30, -40]
        ], dtype=np.int32),
        'shapes': np.array([
            [2, 3],
            [2, 4]
        ], dtype=np.int64),
        'concat_dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Negative concat_dim, int64 values
    input_dict_5 = {
        'name': 'concat_negative_dim_int64',
        'indices': np.array([
            [[0, 2], [1, 0]],
            [[0, 1], [0, 2]]
        ], dtype=np.int64),
        'values': np.array([
            [10, 20],
            [40, 50]
        ], dtype=np.int64),
        'shapes': np.array([
            [2, 3],
            [2, 4]
        ], dtype=np.int64),
        'concat_dim': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D tensors, concat_dim=1, float64 values
    input_dict_6 = {
        'name': 'concat_3d_dim1_float64',
        'indices': np.array([
            [[0, 1, 2], [1, 2, 3]],
            [[0, 0, 0], [1, 1, 1]]
        ], dtype=np.int64),
        'values': np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=np.float64),
        'shapes': np.array([
            [2, 3, 4],
            [2, 2, 4]
        ], dtype=np.int64),
        'concat_dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D tensors, concat_dim=2
    input_dict_7 = {
        'name': 'concat_3d_dim2_string',
        'indices': np.array([
            [[0, 1, 2], [1, 2, 3]],
            [[0, 0, 0], [1, 1, 1]]
        ], dtype=np.int64),
        'values': np.array([
            [b'x', b'y'],
            [b'z', b'w']
        ], dtype=object),
        'shapes': np.array([
            [2, 3, 4],
            [2, 3, 2]
        ], dtype=np.int64),
        'concat_dim': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Higher rank (4D) tensors
    input_dict_8 = {
        'name': 'concat_4d_dim3',
        'indices': np.array([
            [[0, 0, 1, 2]],
            [[0, 0, 0, 0]],
            [[0, 1, 1, 1]]
        ], dtype=np.int64),
        'values': np.array([
            [100],
            [300],
            [400]
        ], dtype=np.int32),
        'shapes': np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 1],
            [1, 2, 3, 2]
        ], dtype=np.int64),
        'concat_dim': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All inputs are empty
    input_dict_9 = {
        'name': 'concat_all_empty',
        'indices': np.empty(shape=(2, 0, 2), dtype=np.int64),
        'values': np.empty(shape=(2, 0), dtype=np.float32),
        'shapes': np.array([
            [2, 3],
            [2, 4]
        ], dtype=np.int64),
        'concat_dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Using boolean values
    input_dict_10 = {
        'name': 'concat_bool_values',
        'indices': np.array([
            [[0, 0]],
            [[1, 1]]
        ], dtype=np.int64),
        'values': np.array([
            [True],
            [False]
        ], dtype=np.bool_),
        'shapes': np.array([
            [3, 2],
            [3, 3]
        ], dtype=np.int64),
        'concat_dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 1D tensors (vectors)
    input_dict_11 = {
        'name': 'concat_1d_vectors_int16',
        'indices': np.array([
            [[1], [3]],
            [[0], [2]]
        ], dtype=np.int64),
        'values': np.array([
            [10, 30],
            [5, 25]
        ], dtype=np.int16),
        'shapes': np.array([
            [5],
            [4]
        ], dtype=np.int64),
        'concat_dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseConcat"] = tf_raw_ops_sparsecancat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseConcat'.")

check_valid('tf.raw_ops.SparseConcat', generated_inputs['tf.raw_ops.SparseConcat'], lib="tf", suffix=0)
