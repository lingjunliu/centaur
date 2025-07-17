
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dynamic_stitch_inputs():
    list_of_inputs = []

    # Input 1: Simple case with scalar indices
    indices = [np.array(0, dtype=np.int32), np.array(1, dtype=np.int32)]
    data = [np.array([10], dtype=np.int32), np.array([20], dtype=np.int32)]
    input_dict = {"name": "simple_case", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector indices
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([10, 12], dtype=np.int32), np.array([11, 13], dtype=np.int32)]
    input_dict = {"name": "vector_indices", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix indices, higher rank data
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32), np.array([[4, 5], [6, 7]], dtype=np.int32)]
    data = [np.array([[[10, 11], [12, 13]], [[14, 15], [16, 17]]], dtype=np.int32),
            np.array([[[20, 21], [22, 23]], [[24, 25], [26, 27]]], dtype=np.int32)]
    input_dict = {"name": "matrix_indices", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data types (float32)
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([10.0, 12.0], dtype=np.float32), np.array([11.0, 13.0], dtype=np.float32)]
    input_dict = {"name": "float_data", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Non-contiguous indices
    indices = [np.array([5, 7], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([50, 70], dtype=np.int32), np.array([10, 30], dtype=np.int32)]
    input_dict = {"name": "non_contiguous", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D indices and corresponding data
    indices = [np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)]
    data = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)]
    input_dict = {"name": "3d_indices", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: indices with zeros
    indices = [np.array([0, 0], dtype=np.int32), np.array([1, 2], dtype=np.int32)]
    data = [np.array([10, 11], dtype=np.int32), np.array([20, 21], dtype=np.int32)]
    input_dict = {"name": "zero_indices", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different shapes for data[i] after indices[i].shape
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([5, 6], dtype=np.int32)]
    input_dict = {"name": "diff_shapes", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Example from the documentation
    indices = [np.array(6, dtype=np.int32), np.array([4, 1], dtype=np.int32), np.array([[5, 2], [0, 3]], dtype=np.int32)]
    data = [np.array([61, 62], dtype=np.int32), np.array([[41, 42], [11, 12]], dtype=np.int32), np.array([[[51, 52], [21, 22]], [[1, 2], [31, 32]]], dtype=np.int32)]
    input_dict = {"name": "doc_example", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty data
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array([], dtype=np.int32), np.array([], dtype=np.int32)]
    input_dict = {"name": "empty_data", "indices": indices, "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DynamicStitch"] = tf_raw_ops_dynamic_stitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DynamicStitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DynamicStitch'.")

check_valid('tf.raw_ops.DynamicStitch', generated_inputs['tf.raw_ops.DynamicStitch'], lib="tf", suffix=0)
