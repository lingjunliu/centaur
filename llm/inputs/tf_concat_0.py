
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_concat_inputs():
    """
    Generates a list of valid inputs for the tf.concat function.
    """
    list_of_inputs = []

    # Input 1: Basic 2D concatenation along axis 0
    t1_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t1_2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
    input_dict_1 = {
        'values': np.array([t1_1, t1_2]),
        'axis': 0,
        'name': 'concat_2d_axis_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 2D concatenation along axis 1
    t2_1 = np.array([[1, 2], [4, 5]], dtype=np.int32)
    t2_2 = np.array([[7, 8], [10, 11]], dtype=np.int32)
    input_dict_2 = {
        'values': np.array([t2_1, t2_2]),
        'axis': 1,
        'name': 'concat_2d_axis_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Concatenating 1D vectors (tensors)
    t3_1 = np.array([1, 2, 3], dtype=np.int32)
    t3_2 = np.array([4, 5, 6], dtype=np.int32)
    input_dict_3 = {
        'values': np.array([t3_1, t3_2]),
        'axis': 0,
        'name': 'concat_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Concatenating 3D tensors along axis 0
    t4_1 = np.random.rand(2, 3, 4).astype(np.float32)
    t4_2 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict_4 = {
        'values': np.array([t4_1, t4_2]),
        'axis': 0,
        'name': 'concat_3d_axis_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Concatenating 3D tensors along axis 2
    t5_1 = np.ones((2, 3, 4), dtype=np.int16)
    t5_2 = np.zeros((2, 3, 4), dtype=np.int16)
    input_dict_5 = {
        'values': np.array([t5_1, t5_2]),
        'axis': 2,
        'name': 'concat_3d_axis_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using a negative axis (-1) for 3D tensors
    t6_1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    t6_2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    input_dict_6 = {
        'values': np.array([t6_1, t6_2]),
        'axis': -1,
        'name': 'concat_3d_negative_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Concatenating more than two tensors
    t7_1 = np.array([[1], [2]], dtype=np.int32)
    t7_2 = np.array([[3], [4]], dtype=np.int32)
    t7_3 = np.array([[5], [6]], dtype=np.int32)
    t7_4 = np.array([[7], [8]], dtype=np.int32)
    input_dict_7 = {
        'values': np.array([t7_1, t7_2, t7_3, t7_4]),
        'axis': 1,
        'name': 'concat_multiple_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using a negative axis (-2)
    t8_1 = np.arange(12).reshape(2, 3, 2).astype(np.float64)
    t8_2 = np.arange(12, 24).reshape(2, 3, 2).astype(np.float64)
    input_dict_8 = {
        'values': np.array([t8_1, t8_2]),
        'axis': -2,
        'name': 'concat_negative_axis_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Concatenating boolean tensors
    t9_1 = np.array([[True, False], [False, True]])
    t9_2 = np.array([[False, False], [True, True]])
    input_dict_9 = {
        'values': np.array([t9_1, t9_2]),
        'axis': 0,
        'name': 'concat_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Concatenating 4D tensors
    t10_1 = np.zeros((1, 2, 3, 4), dtype=np.uint8)
    t10_2 = np.ones((1, 2, 3, 4), dtype=np.uint8)
    input_dict_10 = {
        'values': np.array([t10_1, t10_2]),
        'axis': 3,
        'name': 'concat_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Concatenating tensors with a dimension of size 1
    t11_1 = np.zeros((2, 1, 3), dtype=np.float32)
    t11_2 = np.ones((2, 1, 3), dtype=np.float32)
    input_dict_11 = {
        'values': np.array([t11_1, t11_2]),
        'axis': 1,
        'name': 'concat_dim_one'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Concatenating higher-rank tensors (5D)
    t12_1 = np.ones((2, 2, 2, 2, 2), dtype=np.int8)
    t12_2 = np.zeros((2, 2, 2, 2, 2), dtype=np.int8)
    input_dict_12 = {
        'values': np.array([t12_1, t12_2]),
        'axis': 2,
        'name': 'concat_5d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.concat"] = get_tf_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.concat'.")

check_valid('tf.concat', generated_inputs['tf.concat'], lib="tf", suffix=0)
