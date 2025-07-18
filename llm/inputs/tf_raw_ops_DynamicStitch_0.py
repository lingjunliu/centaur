
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_dynamic_stitch_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DynamicStitch function.
    To work around a limitation in the testing harness that cannot handle ragged
    tensor lists, all tensors within the 'indices' and 'data' lists are
    constrained to have the same shape. This allows them to be stacked into a
    single numpy N-D array, which the harness can process.
    """
    list_of_inputs = []

    # Case 1: Simple case, all tensors in lists have same shape.
    input_dict_1 = {
        'name': 'simple_uniform_shape',
        'indices': np.array([np.array([0, 2], dtype=np.int32), np.array([3, 1], dtype=np.int32)]),
        'data': np.array([np.array([0.0, 2.0], dtype=np.float32), np.array([3.0, 1.0], dtype=np.float32)])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Stitching 2D data slices.
    input_dict_2 = {
        'name': 'stitch_2d_slices_uniform_shape',
        'indices': np.array([np.array([1, 0], dtype=np.int32), np.array([3, 2], dtype=np.int32)]),
        'data': np.array([
            np.array([[10.1, 11.2], [0.3, 1.4]], dtype=np.float64),
            np.array([[30.5, 31.6], [20.5, 21.6]], dtype=np.float64)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Overlapping indices.
    input_dict_3 = {
        'name': 'overlapping_indices_uniform_shape',
        'indices': np.array([np.array([0, 1, 2], dtype=np.int32), np.array([1, 3, 0], dtype=np.int32)]),
        'data': np.array([
            np.array([-1, -2, -3], dtype=np.int32),
            np.array([100, 300, 1000], dtype=np.int32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Single tensor in list.
    input_dict_4 = {
        'name': 'single_tensor_pair',
        'indices': np.array([np.array([[0, 1], [3, 2]], dtype=np.int32)]),
        'data': np.array([np.array([[10, 20], [40, 30]], dtype=np.int64)])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Sparse/non-contiguous indices.
    input_dict_5 = {
        'name': 'sparse_indices_uniform_shape',
        'indices': np.array([np.array([10, 20], dtype=np.int32), np.array([0, 5], dtype=np.int32)]),
        'data': np.array([
            np.array([[10.0, 10.1], [20.0, 20.1]], dtype=np.float32),
            np.array([[0.0, 0.1], [5.0, 5.1]], dtype=np.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Higher dimensional indices and data.
    input_dict_6 = {
        'name': 'high_dim_uniform_shape',
        'indices': np.array([
            np.array([[0, 1], [2, 3]], dtype=np.int32),
            np.array([[4, 5], [6, 7]], dtype=np.int32)
        ]),
        'data': np.array([
            np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.int32),
            np.array([[[5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8]]], dtype=np.int32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: List of empty tensors.
    input_dict_7 = {
        'name': 'list_of_empty_tensors',
        'indices': np.array([np.array([], dtype=np.int32), np.array([], dtype=np.int32)]),
        'data': np.array([np.empty((0, 5), dtype=np.float32), np.empty((0, 5), dtype=np.float32)])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Bool data type
    input_dict_8 = {
        'name': 'bool_data_uniform_shape',
        'indices': np.array([np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]),
        'data': np.array([
            np.array([True, False], dtype=bool),
            np.array([False, True], dtype=bool)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: More tensors in the list (3 tensors)
    input_dict_9 = {
        'name': 'three_tensors_uniform_shape',
        'indices': np.array([
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32)
        ]),
        'data': np.array([
            np.array([0.1, 1.1], dtype=np.float32),
            np.array([2.1, 3.1], dtype=np.float32),
            np.array([4.1, 5.1], dtype=np.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: string data type
    input_dict_10 = {
        'name': 'string_data_uniform_shape',
        'indices': np.array([np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]),
        'data': np.array([np.array(['a', 'c'], dtype=object), np.array(['b', 'd'], dtype=object)])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DynamicStitch"] = get_tf_raw_ops_dynamic_stitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DynamicStitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DynamicStitch'.")

check_valid('tf.raw_ops.DynamicStitch', generated_inputs['tf.raw_ops.DynamicStitch'], lib="tf", suffix=0)
