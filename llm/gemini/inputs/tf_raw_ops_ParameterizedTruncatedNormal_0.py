
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ParameterizedTruncatedNormal_inputs():
    list_of_inputs = []

    # Input 1: Float32, 1D shape, all scalar parameters
    shape_1 = np.array([5], dtype=np.int32)
    means_1 = np.array(0.0, dtype=np.float32)
    stdevs_1 = np.array(1.0, dtype=np.float32)
    minvals_1 = np.array(-2.0, dtype=np.float32)
    maxvals_1 = np.array(2.0, dtype=np.float32)
    input_dict_1 = {
        'seed': 42,
        'seed2': 1,
        'name': "truncated_1",
        'shape': shape_1,
        'means': means_1,
        'stdevs': stdevs_1,
        'minvals': minvals_1,
        'maxvals': maxvals_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Float32, 2D shape, all vector parameters matching batch size 3
    shape_2 = np.array([3, 4], dtype=np.int32)
    means_2 = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    stdevs_2 = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    minvals_2 = np.array([-1.0, 0.0, -3.0], dtype=np.float32)
    maxvals_2 = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    input_dict_2 = {
        'seed': 43,
        'seed2': 123,
        'name': "truncated_2",
        'shape': shape_2,
        'means': means_2,
        'stdevs': stdevs_2,
        'minvals': minvals_2,
        'maxvals': maxvals_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Float64, 3D shape, all scalar parameters, int64 shape
    shape_3 = np.array([2, 2, 2], dtype=np.int64)
    means_3 = np.array(5.0, dtype=np.float64)
    stdevs_3 = np.array(0.1, dtype=np.float64)
    minvals_3 = np.array(4.5, dtype=np.float64)
    maxvals_3 = np.array(5.5, dtype=np.float64)
    input_dict_3 = {
        'seed': 1,
        'seed2': 2,
        'name': "truncated_3",
        'shape': shape_3,
        'means': means_3,
        'stdevs': stdevs_3,
        'minvals': minvals_3,
        'maxvals': maxvals_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Float16, 2D shape, all vector parameters matching batch size 4
    shape_4 = np.array([4, 1], dtype=np.int32)
    means_4 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float16)
    stdevs_4 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float16)
    minvals_4 = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float16)
    maxvals_4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    input_dict_4 = {
        'seed': 44,
        'seed2': 12,
        'name': "truncated_4",
        'shape': shape_4,
        'means': means_4,
        'stdevs': stdevs_4,
        'minvals': minvals_4,
        'maxvals': maxvals_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Float32, 2D shape, all vector parameters matching batch size 2
    shape_5 = np.array([2, 3], dtype=np.int32)
    means_5 = np.array([-1.0, 1.0], dtype=np.float32)
    stdevs_5 = np.array([1.0, 1.0], dtype=np.float32)
    minvals_5 = np.array([-5.0, -5.0], dtype=np.float32)
    maxvals_5 = np.array([0.0, 5.0], dtype=np.float32)
    input_dict_5 = {
        'seed': 7,
        'seed2': 8,
        'name': "truncated_5",
        'shape': shape_5,
        'means': means_5,
        'stdevs': stdevs_5,
        'minvals': minvals_5,
        'maxvals': maxvals_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Float32, 1D shape, large finite bounds
    shape_6 = np.array([3], dtype=np.int32)
    means_6 = np.array(0.0, dtype=np.float32)
    stdevs_6 = np.array(1.0, dtype=np.float32)
    minvals_6 = np.array(-1000.0, dtype=np.float32)
    maxvals_6 = np.array(1000.0, dtype=np.float32)
    input_dict_6 = {
        'seed': 10,
        'seed2': 20,
        'name': "truncated_6",
        'shape': shape_6,
        'means': means_6,
        'stdevs': stdevs_6,
        'minvals': minvals_6,
        'maxvals': maxvals_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Float32, 4D shape (int64), all scalar parameters
    shape_7 = np.array([2, 2, 2, 2], dtype=np.int64)
    means_7 = np.array(10.0, dtype=np.float32)
    stdevs_7 = np.array(5.0, dtype=np.float32)
    minvals_7 = np.array(0.0, dtype=np.float32)
    maxvals_7 = np.array(20.0, dtype=np.float32)
    input_dict_7 = {
        'seed': 99,
        'seed2': 99,
        'name': "truncated_7",
        'shape': shape_7,
        'means': means_7,
        'stdevs': stdevs_7,
        'minvals': minvals_7,
        'maxvals': maxvals_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Float64, 1D shape, all vector parameters matching batch size 5
    shape_8 = np.array([5], dtype=np.int32)
    means_8 = np.array([1., 2., 3., 4., 5.], dtype=np.float64)
    stdevs_8 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    minvals_8 = np.array([0., 0., 0., 0., 0.], dtype=np.float64)
    maxvals_8 = np.array([10., 10., 10., 10., 10.], dtype=np.float64)
    input_dict_8 = {
        'seed': 42,
        'seed2': 42,
        'name': "truncated_8",
        'shape': shape_8,
        'means': means_8,
        'stdevs': stdevs_8,
        'minvals': minvals_8,
        'maxvals': maxvals_8
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Float32, 1D shape, reasonable standard deviation, scalar parameters
    shape_9 = np.array([2], dtype=np.int32)
    means_9 = np.array(0.0, dtype=np.float32)
    stdevs_9 = np.array(0.5, dtype=np.float32)
    minvals_9 = np.array(-1.0, dtype=np.float32)
    maxvals_9 = np.array(1.0, dtype=np.float32)
    input_dict_9 = {
        'seed': 123,
        'seed2': 456,
        'name': "truncated_9",
        'shape': shape_9,
        'means': means_9,
        'stdevs': stdevs_9,
        'minvals': minvals_9,
        'maxvals': maxvals_9
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Float32, 3D shape, all vector parameters matching batch size 4
    shape_10 = np.array([4, 2, 2], dtype=np.int32)
    means_10 = np.array([-10.0, 0.0, 10.0, 20.0], dtype=np.float32)
    stdevs_10 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    minvals_10 = np.array([-100.0, -100.0, -100.0, -100.0], dtype=np.float32)
    maxvals_10 = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    input_dict_10 = {
        'seed': 111,
        'seed2': 222,
        'name': "truncated_10",
        'shape': shape_10,
        'means': means_10,
        'stdevs': stdevs_10,
        'minvals': minvals_10,
        'maxvals': maxvals_10
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ParameterizedTruncatedNormal"] = tf_raw_ops_ParameterizedTruncatedNormal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ParameterizedTruncatedNormal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParameterizedTruncatedNormal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ParameterizedTruncatedNormal', generated_inputs['tf.raw_ops.ParameterizedTruncatedNormal'], lib="tf", suffix=0)
