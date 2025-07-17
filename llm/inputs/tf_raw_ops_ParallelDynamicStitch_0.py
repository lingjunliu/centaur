
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_parallel_dynamic_stitch_inputs():
    list_of_inputs = []

    def make_input(indices, data):
        return {"indices": indices, "data": data, "name": None}

    # Input 1: Basic test case
    indices = [np.array([0, 2, 1], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
    data = [np.array([10, 12, 11], dtype=np.int32), np.array([13, 14], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))

    # Input 2: 2D data
    indices = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
    data = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))

    # Input 3: Scalar indices
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([10], dtype=np.int32), np.array([11], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))

    # Input 4: Different data types
    indices = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
    data = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))

    # Input 5: Empty data, but keeping the dimension
    indices = [np.array([], dtype=np.int32).reshape((0,)), np.array([], dtype=np.int32).reshape((0,))]
    data = [np.array([], dtype=np.int32).reshape((0,)), np.array([], dtype=np.int32).reshape((0,))]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))
    
    # Input 6: Adding more tests and fixing the shape problems
    indices = [np.array([0, 1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]
    data = [np.array([10, 11, 12], dtype=np.int32), np.array([13], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))

    # Input 7: 2D Indices
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32), np.array([[4, 5]], dtype=np.int32)]
    data = [np.array([[[10]], [[11]], [[12]], [[13]]], dtype=np.int32), np.array([[[14]], [[15]]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))
        
    # Input 8: Larger indices
    indices = [np.array([0, 100], dtype=np.int32), np.array([200], dtype=np.int32)]
    data = [np.array([10, 11], dtype=np.int32), np.array([12], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))

    # Input 9: 3D data with consistent shape
    indices = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
    data = [np.array([[[1, 2]], [[3, 4]]], dtype=np.int32), np.array([[[5, 6]], [[7, 8]]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))

    # Input 10: Multiple dimensions and different data type
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), np.array([[5.0, 6.0]], dtype=np.float64)]
    list_of_inputs.append(copy.deepcopy(make_input(indices, data)))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ParallelDynamicStitch"] = tf_raw_ops_parallel_dynamic_stitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ParallelDynamicStitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParallelDynamicStitch'.")

check_valid('tf.raw_ops.ParallelDynamicStitch', generated_inputs['tf.raw_ops.ParallelDynamicStitch'], lib="tf", suffix=0)
