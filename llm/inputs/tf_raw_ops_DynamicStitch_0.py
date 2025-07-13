
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
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector indices, 1D data
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([10, 12], dtype=np.int32), np.array([11, 13], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix indices, 2D data
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32), np.array([[4, 5], [6, 7]], dtype=np.int32)]
    data = [np.array([[10, 11], [12, 13]], dtype=np.int32), np.array([[14, 15], [16, 17]], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes but compatible
    indices = [np.array(0, dtype=np.int32), np.array([1, 2], dtype=np.int32)]
    data = [np.array([10, 11], dtype=np.int32), np.array([[20, 21], [22, 23]], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Simple case with scalar indices and float32 data
    indices = [np.array(0, dtype=np.int32), np.array(1, dtype=np.int32)]
    data = [np.array([10.5], dtype=np.float32), np.array([20.5], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Vector indices, 1D data and float32
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([10.5, 12.5], dtype=np.float32), np.array([11.5, 13.5], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Simple case with scalar indices and string data
    indices = [np.array(0, dtype=np.int32), np.array(1, dtype=np.int32)]
    data = [np.array(["a"], dtype=np.string_), np.array(["b"], dtype=np.string_)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Vector indices, 1D data and string
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array(["a", "c"], dtype=np.string_), np.array(["b", "d"], dtype=np.string_)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element lists
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([10], dtype=np.int32), np.array([20], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different dtypes
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([10.0], dtype=np.float64), np.array([20.0], dtype=np.float64)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: More than 2 indices/data tensors
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array([10], dtype=np.int32), np.array([20], dtype=np.int32), np.array([30], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 12: Mixed shapes for indices and data, ensuring compatibility.
    indices = [np.array(0, dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([10], dtype=np.int32), np.array([20], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Complex numbers
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([1+1j], dtype=np.complex128), np.array([2+2j], dtype=np.complex128)]
    input_dict = {"indices": indices, "data": data, "name": None}
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
