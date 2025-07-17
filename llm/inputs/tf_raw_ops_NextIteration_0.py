
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_next_iteration_inputs():
    list_of_inputs = []

    # Input 1: Integer tensor
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    name = "iteration_1"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    name = "iteration_2"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean tensor
    data = np.array([True, False, True, False, True], dtype=np.bool_)
    name = "iteration_3"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String tensor
    data = np.array([b"a", b"b", b"c", b"d", b"e"], dtype=np.object_)
    name = "iteration_4"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Integer tensor
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    name = "iteration_5"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Float tensor
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "iteration_6"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor
    data = np.array([], dtype=np.int32)
    name = "iteration_7"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large integer tensor
    data = np.array([2**31 - 1, -(2**31)], dtype=np.int32)
    name = "iteration_8"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shape
    data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.int32)
    name = "iteration_9"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex64
    data = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    name = "iteration_10"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NextIteration"] = tf_raw_ops_next_iteration_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NextIteration' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NextIteration'.")

check_valid('tf.raw_ops.NextIteration', generated_inputs['tf.raw_ops.NextIteration'], lib="tf", suffix=0)
