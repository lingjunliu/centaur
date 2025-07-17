
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_parallel_concat_inputs():
    list_of_inputs = []

    # Input 1: Basic example with integers
    values = [np.array([[1, 2]], dtype=np.int32), np.array([[3, 4]], dtype=np.int32), np.array([[5, 6]], dtype=np.int32)]
    shape = [3, 2]
    input_dict = {"values": values, "shape": shape, "name": "concat_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float values
    values = [np.array([[1.5, 2.5]], dtype=np.float32), np.array([[3.5, 4.5]], dtype=np.float32)]
    shape = [2, 2]
    input_dict = {"values": values, "shape": shape, "name": "concat_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    values = [np.array([[-1, -2]], dtype=np.int32), np.array([[-3, -4]], dtype=np.int32)]
    shape = [2, 2]
    input_dict = {"values": values, "shape": shape, "name": "concat_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shape
    values = [np.array([[1, 2, 3]], dtype=np.int32), np.array([[4, 5, 6]], dtype=np.int32)]
    shape = [2, 3]
    input_dict = {"values": values, "shape": shape, "name": "concat_diffshape"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More values
    values = [np.array([[1]], dtype=np.int32), np.array([[2]], dtype=np.int32), np.array([[3]], dtype=np.int32), np.array([[4]], dtype=np.int32)]
    shape = [4, 1]
    input_dict = {"values": values, "shape": shape, "name": "concat_morevalues"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bool values
    values = [np.array([[True, False]], dtype=np.bool_), np.array([[False, True]], dtype=np.bool_)]
    shape = [2, 2]
    input_dict = {"values": values, "shape": shape, "name": "concat_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64
    values = [np.array([[10000000000]], dtype=np.int64), np.array([[20000000000]], dtype=np.int64)]
    shape = [2, 1]
    input_dict = {"values": values, "shape": shape, "name": "concat_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64
    values = [np.array([[1.7976931348623157e+308]], dtype=np.float64), np.array([[2.2250738585072014e-308]], dtype=np.float64)]
    shape = [2, 1]
    input_dict = {"values": values, "shape": shape, "name": "concat_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex64
    values = [np.array([[1+1j]], dtype=np.complex64), np.array([[2+2j]], dtype=np.complex64)]
    shape = [2, 1]
    input_dict = {"values": values, "shape": shape, "name": "concat_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: Complex128
    values = [np.array([[1+1j]], dtype=np.complex128), np.array([[2+2j]], dtype=np.complex128)]
    shape = [2, 1]
    input_dict = {"values": values, "shape": shape, "name": "concat_complex128"}
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
