
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_parallel_concat_inputs():
    list_of_inputs = []

    # Input 1: Valid case with 1D tensors
    values = [np.array([[1]], dtype=np.int32), np.array([[2]], dtype=np.int32), np.array([[3]], dtype=np.int32)]
    shape = np.array([3, 1], dtype=np.int32)
    name = "concat_1d"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid case with 2D tensors
    values = [np.array([[1, 2]], dtype=np.float32), np.array([[3, 4]], dtype=np.float32), np.array([[5, 6]], dtype=np.float32)]
    shape = np.array([3, 2], dtype=np.int32)
    name = "concat_2d"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid case with 3D tensors
    values = [np.array([[[1, 2, 3]]], dtype=np.int64), np.array([[[4, 5, 6]]], dtype=np.int64), np.array([[[7, 8, 9]]], dtype=np.int64)]
    shape = np.array([3, 1, 3], dtype=np.int32)
    name = "concat_3d"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid case with different dtype (float64)
    values = [np.array([[1.1]], dtype=np.float64), np.array([[2.2]], dtype=np.float64), np.array([[3.3]], dtype=np.float64)]
    shape = np.array([3, 1], dtype=np.int32)
    name = "concat_float64"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid case with more tensors
    values = [np.array([[i]], dtype=np.int32) for i in range(1, 6)]
    shape = np.array([5, 1], dtype=np.int32)
    name = "concat_many"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid case with negative values
    values = [np.array([[-1]], dtype=np.int32), np.array([[-2]], dtype=np.int32), np.array([[-3]], dtype=np.int32)]
    shape = np.array([3, 1], dtype=np.int32)
    name = "concat_negative"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Valid case with shape specified as tuple
    values = [np.array([[1]], dtype=np.int32), np.array([[2]], dtype=np.int32), np.array([[3]], dtype=np.int32)]
    shape = np.array([3, 1], dtype=np.int32)
    name = "concat_1d_tuple"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid case with complex numbers
    values = [np.array([[1+1j]], dtype=np.complex64), np.array([[2+2j]], dtype=np.complex64), np.array([[3+3j]], dtype=np.complex64)]
    shape = np.array([3, 1], dtype=np.int32)
    name = "concat_complex"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex shapes
    values = [np.array([[1, 2, 3]], dtype=np.int32), np.array([[4, 5, 6]], dtype=np.int32)]
    shape = np.array([2, 3], dtype=np.int32)
    name = "concat_wider"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different type for shape (list of integers). Previously used tuples.
    values = [np.array([[1, 2]], dtype=np.float32), np.array([[3, 4]], dtype=np.float32)]
    shape = np.array([2, 2], dtype=np.int32)
    name = "concat_2d_ints"
    input_dict = {"values": values, "shape": shape.tolist(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ParallelConcat"] = tf_raw_ops_parallel_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ParallelConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParallelConcat'.")

check_valid('tf.raw_ops.ParallelConcat', generated_inputs['tf.raw_ops.ParallelConcat'], lib="tf", suffix=0)
